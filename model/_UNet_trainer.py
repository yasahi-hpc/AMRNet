from ._base_trainer import _BaseTrainer, MeasureMemory
import pathlib
import torch.multiprocessing as mp
import torch
from torch import nn
import horovod.torch as hvd
import numpy as np
import xarray as xr
import itertools
from .flow_dataset import FlowDataset
from .unet import UNet

class UNetTrainer(_BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'UNet'

    def _initialize(self, **kwargs):
        # Horovod: Initialize library
        hvd.init()
        torch.manual_seed(self.seed)

        if self.device == 'cuda':
            # Horovod: Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(self.seed)

        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)
        
        self.rank, self.size = hvd.rank(), hvd.size()
        self.master = self.rank == 0

        super()._prepare_dirs()
        self.train_loader, self.val_loader, self.test_loader = super()._dataloaders()

        self.model = self._get_model(self.run_number)
        self.model = self.model.to(self.device)

        ## Optimizers
        # By default, Adasum doesn't need scaling up leraning rate
        lr_scaler = hvd.size() if not self.use_adasum else 1
        if self.device == 'cuda' and self.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()
        lr = self.lr * lr_scaler
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(self.beta_1, self.beta_2))

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.opt, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.opt = hvd.DistributedOptimizer(self.opt,
                                            named_parameters=self.model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Adasum if self.use_adasum else hvd.Average,
                                            gradient_predivide_factor=self.gradient_predivide_factor)

        self.criterion = nn.L1Loss() if self.loss_type == 'mae_loss' else nn.MSELoss(reduction='mean')

        # Set normalization coefficients
        super()._set_normalization_coefs(shape=[1,-1,1,1])

        # Memory measurement
        device_name = 'cpu'
        if self.device == 'cuda':
            local_rank = hvd.local_rank()
            device_name = f'{self.device}:{local_rank}'
        self.memory = MeasureMemory(device=device_name)

        # Synchronize
        if self.device == 'cuda':
            torch.cuda.synchronize() # Waits for everything to finish running

    def _get_model(self, run_number):
        model = UNet(n_layers=8, hidden_dim=16, dim=self.dim)

        self.epoch_start = 0
        if run_number > 0:
            if self.master:
                print(f'restart, {run_number}')
            # Load model states from previous run
            prev_run_number = run_number - 1
            prev_result_filename = self.out_dir / f'flow_cnn_result_rank{rank}_rst{prev_run_number:03}.h5'

            if not prev_result_filename.is_file():
                raise IOError(f'prev_result_filename')

            ds_prev = xr.open_dataset(path=prev_result_filename, engine='netcdf4')

            # To load the previous files
            epoch_end = ds_prev.attrs['epoch_end']
            model.load_state_dict( torch.load(f'{self.model_dir}/model_{rank}_{epoch_end:03}.pt') )

            # Next epoch should start from epoch_end + 1
            self.epoch_start = epoch_end + 1

        return model
    
    def _save_models(self, total_epoch):
        torch.save(self.model.state_dict(), f'{self.model_dir}/model_{self.rank}_{total_epoch:03}.pt')
        
    def _step(self, epoch):
        total_epoch = self.epoch_start + epoch
        if self.master:
            print(f'Epoch {total_epoch}')

        self.train_sampler.set_epoch(total_epoch)
        self.val_sampler.set_epoch(total_epoch)
        self.test_sampler.set_epoch(total_epoch)

        # Training
        with torch.enable_grad():
            self._train(data_loader=self.val_loader, epoch=total_epoch)
            #self._train(data_loader=self.train_loader, epoch=total_epoch)

        # Validation
        with torch.no_grad():
            self._validation(data_loader=self.val_loader, epoch=total_epoch, name='validation')

        # Test
        with torch.no_grad():
            self._validation(data_loader=self.test_loader, epoch=total_epoch, name='test')

        # Save models
        if epoch % 10 == 0:
            self._save_models(total_epoch=total_epoch)

    def _finalize(self, seconds):
        # Save models
        total_epoch = self.epoch_start + self.n_epochs - 1
        self._save_models(total_epoch=total_epoch)

        # Saving relevant data in a hdf5 file
        data_vars = {}
        for key, value in self.elapsed_times.items():
            if len(value) > 0:
                var = np.asarray(value)
                nb_calls = len(var) //self. n_epochs
                var = var.reshape((self.n_epochs, nb_calls))
                data_vars[f'seconds_{key}']= (['epochs'], np.sum(var, axis=1))

        # Losses
        for key, value in self.loss_dict.items():
            if len(value) > 0:
                data_vars[key] = (['epochs'], np.asarray(value))

        coords = {'epochs': np.arange(self.n_epochs) + self.epoch_start}
        attrs = super()._get_attrs()
        attrs['seconds'] = seconds

        attrs['memory_reserved'] = self.memory_consumption['reserved']
        attrs['memory_alloc'] = self.memory_consumption['alloc']

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        result_filename = self.out_dir / f'flow_cnn_result_rank{self.rank}_rst{self.run_number:03}.h5'
        ds.to_netcdf(result_filename, engine='netcdf4')

        if self.master:
            log_filename = f'log_rst{self.run_number:03}.txt'
             
            with open(log_filename, 'w') as f:
                f.write( f'It took {seconds} seconds for {self.n_epochs} epochs')

    ########### Main scripts 
    def _train(self, data_loader, epoch):
        name = 'train'
        self.model.train()

        log_loss = 0
        nb_samples = len(data_loader.sampler)

        level = 2
        # Timers
        for sdf, flows in data_loader:
            # Load data and meta-data
            *_, sdf_Lv2 = sdf
            *_, flows_Lv2 = flows

            batch_len = len(sdf_Lv2)

            ## To device
            self.timer.start()

            sdf_Lv2   = sdf_Lv2.to(self.device)
            flows_Lv2 = flows_Lv2.to(self.device)

            self.timer.stop()
            self.elapsed_times[f'MemcpyH2D_{name}'].append(self.timer.elapsed_seconds())

            ## Normalization or standardization
            sdf_Lv2 = super()._preprocess(sdf_Lv2, self.sdf_Lv2_var0, self.sdf_Lv2_var1)
            flows_Lv2 = super()._preprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

            # Objectives: construct pred_flows_Lv2
            pred_flows_Lv2_ = torch.zeros_like(flows_Lv2, device='cpu')

            #### Train Lv2
            self.timer.start()

            ### Update weights
            pred_flows_Lv2 = self.model(sdf_Lv2)
            loss_mae = self.criterion(pred_flows_Lv2, flows_Lv2)

            self.opt.zero_grad()

            ### Measure memory usage before backward
            self.memory.measure()
            if 'reserved' not in self.memory_consumption:
                self.memory_consumption['reserved'] = self.memory.reserved()
                self.memory_consumption['alloc']    = self.memory.alloc()

            loss_mae.backward()
            self.opt.step()

            ### Log losses
            log_loss += loss_mae.item() / nb_samples

            ### Destandardization and save
            pred_flows_Lv2 = super()._postprocess(pred_flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
            pred_flows_Lv2_ = pred_flows_Lv2.detach().cpu()

            self.timer.stop()
            self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            # Saving figures

        # Horovod: average metric values across workers.
        losses = {}
        losses[f'log_loss_{name}_{self.loss_type}_Lv{level}'] = log_loss

        for key, value in losses.items():
            loss = super()._metric_average(value, key)
            self.loss_dict[key].append(loss)

    def _validation(self, data_loader, epoch, name):
        self.model.eval()
        log_loss = 0
        nb_samples = len(data_loader.sampler)

        level = 2
        for sdf, flows in data_loader:
            # Load data and meta-data
            *_, sdf_Lv2 = sdf
            *_, flows_Lv2 = flows

            batch_len = len(sdf_Lv2)

            ## To device
            self.timer.start()

            sdf_Lv2   = sdf_Lv2.to(self.device)
            flows_Lv2 = flows_Lv2.to(self.device)

            self.timer.stop()
            self.elapsed_times[f'MemcpyH2D_{name}'].append(self.timer.elapsed_seconds())

            ## Normalization or standardization
            sdf_Lv2 = super()._preprocess(sdf_Lv2, self.sdf_Lv2_var0, self.sdf_Lv2_var1)
            flows_Lv2 = super()._preprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

            # Objectives: construct pred_flows_Lv2
            pred_flows_Lv2_ = torch.zeros_like(flows_Lv2, device='cpu')

            #### Train Lv0
            self.timer.start()

            ### Update weights
            pred_flows_Lv2 = self.model(sdf_Lv2)
            loss_mae = self.criterion(pred_flows_Lv2, flows_Lv2)

            ### Log losses
            log_loss += loss_mae.item() / nb_samples

            ### Destandardization and save
            pred_flows_Lv2 = super()._postprocess(pred_flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
            pred_flows_Lv2_ = pred_flows_Lv2.detach().cpu()

            self.timer.stop()
            self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            # Saving figures

        # Horovod: average metric values across workers.
        losses = {}
        losses[f'log_loss_{name}_{self.loss_type}_Lv{level}'] = log_loss

        for key, value in losses.items():
            loss = super()._metric_average(value, key)
            self.loss_dict[key].append(loss)
