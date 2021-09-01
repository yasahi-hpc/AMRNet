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
import sys
from .visualization import save_flows
from .converter import save_as_netcdf

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

    def _initialize_for_inference(self, **kwargs):
        # Set output directory
        super()._prepare_dirs()
        
        self.train_loader, self.val_loader, self.test_loader = super()._dataloaders()
        self.model = self._get_model(self.run_number)
        self.model = self.model.to(self.device)
        
        # Set normalization coefficients
        super()._set_normalization_coefs(shape=[1,-1,1,1])
        
        # Memory measurement
        self.memory = MeasureMemory(device=self.device)
        
        # Synchronize
        if self.device == 'cuda':
            torch.cuda.synchronize() # Waits for everything to finish running

    def _get_model(self, run_number):
        model = UNet(n_layers=8, hidden_dim=8, dim=self.dim, padding_mode=self.padding_mode)

        if self.inference_mode:
            self.epoch_start = self.load_nth_state_file
            # To load the state file for inference
            rank = 0
            model.load_state_dict( torch.load(f'{self.state_file_dir}/model_{rank}_{self.epoch_start:03}.pt') )
             
        else:
            self.epoch_start = 0
            if run_number > 0:
                if self.master:
                    print(f'restart, {run_number}')
                # Load model states from previous run
                prev_run_number = run_number - 1
                prev_result_filename = self.out_dir / f'flow_cnn_result_rank{self.rank}_rst{prev_run_number:03}.h5'

                if not prev_result_filename.is_file():
                    raise IOError(f'prev_result_filename')

                ds_prev = xr.open_dataset(prev_result_filename, engine='netcdf4')

                # To load the previous files
                epoch_end = ds_prev.attrs['epoch_end']
                model.load_state_dict( torch.load(f'{self.model_dir}/model_{self.rank}_{epoch_end:03}.pt') )

                # Next epoch should start from epoch_end + 1
                self.epoch_start = int(epoch_end) + 1

        return model
    
    def _save_models(self, total_epoch):
        torch.save(self.model.state_dict(), f'{self.model_dir}/model_{self.rank}_{total_epoch:03}.pt')

    ########### Main scripts 
    def _train(self, data_loader, epoch):
        name = 'train'
        self.model.train()

        log_loss = 0
        nb_samples = len(data_loader.sampler)

        level = 2
        # Timers
        for i, (sdf, flows) in enumerate(data_loader):
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

            # Keep sdfs on CPUs
            sdf_Lv2_cpu = sdf_Lv2.to('cpu')

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
            if i == 0:
                self.timer.start()
                flows_Lv2 = super()._postprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

                ### Zeros inside objects
                pred_flows_Lv2_ = super()._zeros_inside_objects(pred_flows_Lv2_, sdf_Lv2_cpu)

                ### Lv2 figures
                level = 2
                save_flows(flows_Lv2, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv2_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                
                # Check errors
                save_flows(pred_flows_Lv2_-flows_Lv2.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)
                self.timer.stop()
                self.elapsed_times[f'save_figs_{name}'].append(self.timer.elapsed_seconds())

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
        for i, (sdf, flows) in enumerate(data_loader):
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

            # Keep sdfs on CPUs
            sdf_Lv2_cpu = sdf_Lv2.to('cpu')

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
            if i == 0:
                self.timer.start()
                flows_Lv2 = super()._postprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

                ### Zeros inside objects
                pred_flows_Lv2_ = super()._zeros_inside_objects(pred_flows_Lv2_, sdf_Lv2_cpu)

                ### Lv2 figures
                level = 2
                save_flows(flows_Lv2, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv2_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                
                # Check errors
                save_flows(pred_flows_Lv2_-flows_Lv2.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)
                self.timer.stop()
                self.elapsed_times[f'save_figs_{name}'].append(self.timer.elapsed_seconds())

        # Horovod: average metric values across workers.
        losses = {}
        losses[f'log_loss_{name}_{self.loss_type}_Lv{level}'] = log_loss

        for key, value in losses.items():
            loss = super()._metric_average(value, key)
            self.loss_dict[key].append(loss)

    ### For inference
    def _infer(self):
        with torch.no_grad():
            self._convert(data_loader=self.val_loader, name='validation')
            self._convert(data_loader=self.test_loader, name='test')
             
    def _convert(self, data_loader, name):
        self.model.eval()
        level = 2
        for indices, sdf, flows in data_loader:
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
        
            # Keep sdfs on CPUs
            sdf_Lv2_cpu = sdf_Lv2.to('cpu')
        
            ## Normalization or standardization
            sdf_Lv2 = super()._preprocess(sdf_Lv2, self.sdf_Lv2_var0, self.sdf_Lv2_var1)
            flows_Lv2 = super()._preprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
        
            # Objectives: construct pred_flows_Lv2
            pred_flows_Lv2_ = torch.zeros_like(flows_Lv2, device='cpu')
        
            #### Infer Lv2
            self.timer.start()
        
            ### Update weights
            pred_flows_Lv2 = self.model(sdf_Lv2)
        
            ### Destandardization and save
            pred_flows_Lv2 = super()._postprocess(pred_flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
            pred_flows_Lv2_ = pred_flows_Lv2.detach().cpu()
        
            self.timer.stop()
            self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())
        
            # Save the data in netcdf format
            self.timer.start()
            flows_Lv2 = super()._postprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
        
            ### Zeros inside objects
            pred_flows_Lv2_ = super()._zeros_inside_objects(pred_flows_Lv2_, sdf_Lv2_cpu)
        
            ### Lv2 data
            save_as_netcdf(sdf=sdf_Lv2_cpu, real_flows=flows_Lv2.cpu(), pred_flows=pred_flows_Lv2_,
                           indices=indices, epoch=self.epoch_start, level=level, name=name, data_dir=self.inference_dir)
        
            self.timer.stop()
            self.elapsed_times[f'save_data_{name}'].append(self.timer.elapsed_seconds())
