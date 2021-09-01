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
from .visualization import save_flows
from .converter import save_as_netcdf

class PatchedUNetTrainer(_BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'patched_UNet'

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

        self.model0, self.model1, self.model2 = self._get_model(self.run_number)
        self.model0 = self.model0.to(self.device)
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        ## Optimizers
        # By default, Adasum doesn't need scaling up leraning rate
        lr_scaler = hvd.size() if not self.use_adasum else 1
        if self.device == 'cuda' and self.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()
        lr = self.lr * lr_scaler
        self.opt0 = torch.optim.Adam(self.model0.parameters(), lr=lr, betas=(self.beta_1, self.beta_2))
        self.opt1 = torch.optim.Adam(self.model1.parameters(), lr=lr, betas=(self.beta_1, self.beta_2))
        self.opt2 = torch.optim.Adam(self.model2.parameters(), lr=lr, betas=(self.beta_1, self.beta_2))

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.model0.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.model1.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.model2.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.opt0, root_rank=0)
        hvd.broadcast_optimizer_state(self.opt1, root_rank=0)
        hvd.broadcast_optimizer_state(self.opt2, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.opt0 = hvd.DistributedOptimizer(self.opt0,
                                             named_parameters=self.model0.named_parameters(),
                                             compression=compression,
                                             op=hvd.Adasum if self.use_adasum else hvd.Average,
                                             gradient_predivide_factor=self.gradient_predivide_factor)
        self.opt1 = hvd.DistributedOptimizer(self.opt1,
                                             named_parameters=self.model1.named_parameters(),
                                             compression=compression,
                                             op=hvd.Adasum if self.use_adasum else hvd.Average,
                                             gradient_predivide_factor=self.gradient_predivide_factor)
        self.opt2 = hvd.DistributedOptimizer(self.opt2,
                                             named_parameters=self.model2.named_parameters(),
                                             compression=compression,
                                             op=hvd.Adasum if self.use_adasum else hvd.Average,
                                             gradient_predivide_factor=self.gradient_predivide_factor)

        self.criterion = nn.L1Loss() if self.loss_type == 'mae_loss' else nn.MSELoss(reduction='mean')

        # Set normalization coefficients
        super()._set_normalization_coefs(shape=[1,1,1,-1,1,1])

        # Memory measurement
        self.memory = MeasureMemory(device=self.device)

        # Synchronize
        if self.device == 'cuda':
            torch.cuda.synchronize() # Waits for everything to finish running

    def _initialize_for_inference(self, **kwargs):
        # Set output directory
        super()._prepare_dirs()
        
        self.train_loader, self.val_loader, self.test_loader = super()._dataloaders()
        self.model0, self.model1, self.model2 = self._get_model(self.run_number)
        self.model0 = self.model0.to(self.device)
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        
        # Set normalization coefficients
        super()._set_normalization_coefs(shape=[1,1,1,-1,1,1])
        
        # Memory measurement
        self.memory = MeasureMemory(device=self.device)
        
        # Synchronize
        if self.device == 'cuda':
            torch.cuda.synchronize() # Waits for everything to finish running

    def _get_model(self, run_number):
        model0 = UNet(dim=self.dim, padding_mode=self.padding_mode)
        model1 = UNet(dim=self.dim, padding_mode=self.padding_mode)
        model2 = UNet(dim=self.dim, padding_mode=self.padding_mode)

        if self.inference_mode:
            self.epoch_start = self.load_nth_state_file
            # To load the state file for inference
            rank = 0
            model0.load_state_dict( torch.load(f'{self.state_file_dir}/model0_{rank}_{self.epoch_start:03}.pt') )
            model1.load_state_dict( torch.load(f'{self.state_file_dir}/model1_{rank}_{self.epoch_start:03}.pt') )
            model2.load_state_dict( torch.load(f'{self.state_file_dir}/model2_{rank}_{self.epoch_start:03}.pt') )
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
                model0.load_state_dict( torch.load(f'{self.model_dir}/model0_{self.rank}_{epoch_end:03}.pt') )
                model1.load_state_dict( torch.load(f'{self.model_dir}/model1_{self.rank}_{epoch_end:03}.pt') )
                model2.load_state_dict( torch.load(f'{self.model_dir}/model2_{self.rank}_{epoch_end:03}.pt') )

                # Next epoch should start from epoch_end + 1
                self.epoch_start = int(epoch_end) + 1

        return model0, model1, model2
    
    def _save_models(self, total_epoch):
        torch.save(self.model0.state_dict(), f'{self.model_dir}/model0_{self.rank}_{total_epoch:03}.pt')
        torch.save(self.model1.state_dict(), f'{self.model_dir}/model1_{self.rank}_{total_epoch:03}.pt')
        torch.save(self.model2.state_dict(), f'{self.model_dir}/model2_{self.rank}_{total_epoch:03}.pt')

    ########### Main scripts 
    def _train(self, data_loader, epoch):
        name = 'train'
        self.model0.train()
        self.model1.train()
        self.model2.train()

        log_loss = [0] * 3
        nb_samples = len(data_loader.sampler)

        for i, (sdf, flows) in enumerate(data_loader):
            # Load data and meta-data
            sdf_Lv0, sdf_Lv1, sdf_Lv2 = sdf
            flows_Lv0, flows_Lv1, flows_Lv2 = flows

            _, patch_y_Lv1, patch_x_Lv1, *_ = sdf_Lv1.shape
            _, patch_y_Lv2, patch_x_Lv2, *_ = sdf_Lv2.shape

            # Number of sub patches in each level
            nb_patches_Lv2 = patch_y_Lv2 * patch_x_Lv2
            nb_patches_Lv1 = patch_y_Lv1 * patch_x_Lv1
            
            # Sub patch inside the Lv1 patch
            patch_y_Lv2 = patch_y_Lv2 // patch_y_Lv1
            patch_x_Lv2 = patch_x_Lv2 // patch_x_Lv1
            
            batch_len = len(sdf_Lv0)

            ## To device
            self.timer.start()

            sdf_Lv0, sdf_Lv1, sdf_Lv2 = sdf_Lv0.to(self.device), sdf_Lv1.to(self.device), sdf_Lv2.to(self.device)
            flows_Lv0, flows_Lv1, flows_Lv2 = flows_Lv0.to(self.device), flows_Lv1.to(self.device), flows_Lv2.to(self.device)

            self.timer.stop()
            self.elapsed_times[f'MemcpyH2D_{name}'].append(self.timer.elapsed_seconds())

            # Keep sdfs on CPUs
            sdf_Lv0_cpu = sdf_Lv0.to('cpu')
            sdf_Lv1_cpu = sdf_Lv1.to('cpu')
            sdf_Lv2_cpu = sdf_Lv2.to('cpu')

            ## Normalization or standardization
            sdf_Lv0 = super()._preprocess(sdf_Lv0, self.sdf_Lv0_var0, self.sdf_Lv0_var1)
            sdf_Lv1 = super()._preprocess(sdf_Lv1, self.sdf_Lv1_var0, self.sdf_Lv1_var1)
            sdf_Lv2 = super()._preprocess(sdf_Lv2, self.sdf_Lv2_var0, self.sdf_Lv2_var1)

            flows_Lv0 = super()._preprocess(flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            flows_Lv1 = super()._preprocess(flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
            flows_Lv2 = super()._preprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

            # Objectives: construct pred_flows_Lv0-Lv2
            pred_flows_Lv0_ = torch.zeros_like(flows_Lv0, device='cpu')
            pred_flows_Lv1_ = torch.zeros_like(flows_Lv1, device='cpu')
            pred_flows_Lv2_ = torch.zeros_like(flows_Lv2, device='cpu')

            #### Train Lv0
            self.timer.start()
            sdf_Lv0_ = sdf_Lv0[:, 0, 0]
            flows_Lv0_ = flows_Lv0[:, 0, 0]

            ### Update weights
            pred_flows_Lv0 = self.model0(sdf_Lv0_)
            loss_mae = self.criterion(pred_flows_Lv0, flows_Lv0_)

            self.opt0.zero_grad()
            loss_mae.backward()
            self.opt0.step()

            ### Log losses
            level = 0
            log_loss[level] += loss_mae.item() / nb_samples

            ### Destandardization and save
            pred_flows_Lv0 = super()._postprocess(pred_flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            pred_flows_Lv0_[:, 0, 0, :, :, :] = pred_flows_Lv0.detach().cpu()

            self.timer.stop()
            self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            ### Train Lv1
            for iy_Lv1, ix_Lv1 in itertools.product(range(patch_y_Lv1), range(patch_x_Lv1)):
                self.timer.start()
                sdf_Lv1_ = sdf_Lv1[:, iy_Lv1, ix_Lv1]
                flows_Lv1_ = flows_Lv1[:, iy_Lv1, ix_Lv1]

                ### Update weights
                pred_flows_Lv1 = self.model1(sdf_Lv1_)
                loss_mae = self.criterion(pred_flows_Lv1, flows_Lv1_)

                self.opt1.zero_grad()
                loss_mae.backward()
                self.opt1.step()

                ### Log losses
                level = 1
                log_loss[level] += loss_mae.item() / (nb_samples * nb_patches_Lv1)

                pred_flows_Lv1 = super()._postprocess(pred_flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
                pred_flows_Lv1_[:, iy_Lv1, ix_Lv1, :, :, :] = pred_flows_Lv1.detach().cpu()

                self.timer.stop()
                self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

                ### Train Lv2
                for iy_Lv2, ix_Lv2 in itertools.product(range(patch_y_Lv2), range(patch_x_Lv2)):
                    self.timer.start()
                    global_ix_Lv2 = ix_Lv2 + (ix_Lv1 * patch_x_Lv2)
                    global_iy_Lv2 = iy_Lv2 + (iy_Lv1 * patch_y_Lv2)
                    sdf_Lv2_   = sdf_Lv2[:, global_iy_Lv2, global_ix_Lv2]
                    flows_Lv2_ = flows_Lv2[:, global_iy_Lv2, global_ix_Lv2]
                    
                    ### Update generator weights
                    pred_flows_Lv2 = self.model2(sdf_Lv2_)
                    loss_mae = self.criterion(pred_flows_Lv2, flows_Lv2_)
                    
                    self.opt2.zero_grad()

                    ### Measure memory usage before backward
                    self.memory.measure()
                    if 'reserved' not in self.memory_consumption:
                        self.memory_consumption['reserved'] = self.memory.reserved()
                        self.memory_consumption['alloc'] = self.memory.alloc()

                    loss_mae.backward()
                    self.opt2.step()
                    
                    ### Log losses
                    level = 2
                    log_loss[level] += loss_mae.item() / (nb_samples * nb_patches_Lv2)
                    
                    ### Destandardization and save
                    pred_flows_Lv2 = super()._postprocess(pred_flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
                    pred_flows_Lv2_[:, global_iy_Lv2, global_ix_Lv2, :, :, :] = pred_flows_Lv2.detach().cpu()

                    self.timer.stop()
                    self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            # Saving figures
            if i == 0:
                self.timer.start()
                flows_Lv0 = super()._postprocess(flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
                flows_Lv1 = super()._postprocess(flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
                flows_Lv2 = super()._postprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

                ### Zeros inside objects
                pred_flows_Lv0_ = super()._zeros_inside_objects(pred_flows_Lv0_, sdf_Lv0_cpu)
                pred_flows_Lv1_ = super()._zeros_inside_objects(pred_flows_Lv1_, sdf_Lv1_cpu)
                pred_flows_Lv2_ = super()._zeros_inside_objects(pred_flows_Lv2_, sdf_Lv2_cpu)
                
                ### Lv0 figures
                level = 0
                save_flows(flows_Lv0, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv0_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                save_flows(pred_flows_Lv0_-flows_Lv0.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)

                ### Lv1 figures
                level = 1
                save_flows(flows_Lv1, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv1_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                save_flows(pred_flows_Lv1_-flows_Lv1.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)

                ### Lv2 figures
                level = 2
                save_flows(flows_Lv2, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv2_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                save_flows(pred_flows_Lv2_-flows_Lv2.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)
                
                self.timer.stop()
                self.elapsed_times[f'save_figs_{name}'].append(self.timer.elapsed_seconds())

        # Horovod: average metric values across workers.
        losses = {}
        for level in range(3):
            losses[f'log_loss_{name}_{self.loss_type}_Lv{level}'] = log_loss[level]

        for key, value in losses.items():
            loss = super()._metric_average(value, key)
            self.loss_dict[key].append(loss)

    def _validation(self, data_loader, epoch, name):
        self.model0.eval()
        self.model1.eval()
        self.model2.eval()
        log_loss = [0] * 3
        nb_samples = len(data_loader.sampler)

        for i, (sdf, flows) in enumerate(data_loader):
            # Load data and meta-data
            sdf_Lv0, sdf_Lv1, sdf_Lv2 = sdf
            flows_Lv0, flows_Lv1, flows_Lv2 = flows

            _, patch_y_Lv1, patch_x_Lv1, *_ = sdf_Lv1.shape
            _, patch_y_Lv2, patch_x_Lv2, *_ = sdf_Lv2.shape

            # Number of sub patches in each level
            nb_patches_Lv2 = patch_y_Lv2 * patch_x_Lv2
            nb_patches_Lv1 = patch_y_Lv1 * patch_x_Lv1
            
            # Sub patch inside the Lv1 patch
            patch_y_Lv2 = patch_y_Lv2 // patch_y_Lv1
            patch_x_Lv2 = patch_x_Lv2 // patch_x_Lv1
            
            batch_len = len(sdf_Lv0)

            ## To device
            self.timer.start()

            sdf_Lv0, sdf_Lv1, sdf_Lv2 = sdf_Lv0.to(self.device), sdf_Lv1.to(self.device), sdf_Lv2.to(self.device)
            flows_Lv0, flows_Lv1, flows_Lv2 = flows_Lv0.to(self.device), flows_Lv1.to(self.device), flows_Lv2.to(self.device)

            self.timer.stop()
            self.elapsed_times[f'MemcpyH2D_{name}'].append(self.timer.elapsed_seconds())

            # Keep sdfs on CPUs
            sdf_Lv0_cpu = sdf_Lv0.to('cpu')
            sdf_Lv1_cpu = sdf_Lv1.to('cpu')
            sdf_Lv2_cpu = sdf_Lv2.to('cpu')

            ## Normalization or standardization
            sdf_Lv0 = super()._preprocess(sdf_Lv0, self.sdf_Lv0_var0, self.sdf_Lv0_var1)
            sdf_Lv1 = super()._preprocess(sdf_Lv1, self.sdf_Lv1_var0, self.sdf_Lv1_var1)
            sdf_Lv2 = super()._preprocess(sdf_Lv2, self.sdf_Lv2_var0, self.sdf_Lv2_var1)

            flows_Lv0 = super()._preprocess(flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            flows_Lv1 = super()._preprocess(flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
            flows_Lv2 = super()._preprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

            # Objectives: construct pred_flows_Lv0-Lv2
            pred_flows_Lv0_ = torch.zeros_like(flows_Lv0, device='cpu')
            pred_flows_Lv1_ = torch.zeros_like(flows_Lv1, device='cpu')
            pred_flows_Lv2_ = torch.zeros_like(flows_Lv2, device='cpu')

            #### Train Lv0
            self.timer.start()
            sdf_Lv0_ = sdf_Lv0[:, 0, 0]
            flows_Lv0_ = flows_Lv0[:, 0, 0]

            ### Update weights
            pred_flows_Lv0 = self.model0(sdf_Lv0_)
            loss_mae = self.criterion(pred_flows_Lv0, flows_Lv0_)

            ### Log losses
            level = 0
            log_loss[level] += loss_mae.item() / nb_samples

            ### Destandardization and save
            pred_flows_Lv0 = super()._postprocess(pred_flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            pred_flows_Lv0_[:, 0, 0, :, :, :] = pred_flows_Lv0.detach().cpu()

            self.timer.stop()
            self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            ### Train Lv1
            for iy_Lv1, ix_Lv1 in itertools.product(range(patch_y_Lv1), range(patch_x_Lv1)):
                self.timer.start()
                sdf_Lv1_ = sdf_Lv1[:, iy_Lv1, ix_Lv1]
                flows_Lv1_ = flows_Lv1[:, iy_Lv1, ix_Lv1]

                ### Update weights
                pred_flows_Lv1 = self.model1(sdf_Lv1_)
                loss_mae = self.criterion(pred_flows_Lv1, flows_Lv1_)

                ### Log losses
                level = 1
                log_loss[level] += loss_mae.item() / (nb_samples * nb_patches_Lv1)

                pred_flows_Lv1 = super()._postprocess(pred_flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
                pred_flows_Lv1_[:, iy_Lv1, ix_Lv1, :, :, :] = pred_flows_Lv1.detach().cpu()

                self.timer.stop()
                self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

                ### Train Lv2
                for iy_Lv2, ix_Lv2 in itertools.product(range(patch_y_Lv2), range(patch_x_Lv2)):
                    self.timer.start()
                    global_ix_Lv2 = ix_Lv2 + (ix_Lv1 * patch_x_Lv2)
                    global_iy_Lv2 = iy_Lv2 + (iy_Lv1 * patch_y_Lv2)
                    sdf_Lv2_   = sdf_Lv2[:, global_iy_Lv2, global_ix_Lv2]
                    flows_Lv2_ = flows_Lv2[:, global_iy_Lv2, global_ix_Lv2]

                    ### Update generator weights
                    pred_flows_Lv2 = self.model2(sdf_Lv2_)
                    loss_mae = self.criterion(pred_flows_Lv2, flows_Lv2_)
                    
                    ### Log losses
                    level = 2
                    log_loss[level] += loss_mae.item() / (nb_samples * nb_patches_Lv2)
                    
                    ### Destandardization and save
                    pred_flows_Lv2 = super()._postprocess(pred_flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
                    pred_flows_Lv2_[:, global_iy_Lv2, global_ix_Lv2, :, :, :] = pred_flows_Lv2.detach().cpu()

                    self.timer.stop()
                    self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            # Saving figures
            if i == 0:
                self.timer.start()
                flows_Lv0 = super()._postprocess(flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
                flows_Lv1 = super()._postprocess(flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
                flows_Lv2 = super()._postprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)

                ### Zeros inside objects
                pred_flows_Lv0_ = super()._zeros_inside_objects(pred_flows_Lv0_, sdf_Lv0_cpu)
                pred_flows_Lv1_ = super()._zeros_inside_objects(pred_flows_Lv1_, sdf_Lv1_cpu)
                pred_flows_Lv2_ = super()._zeros_inside_objects(pred_flows_Lv2_, sdf_Lv2_cpu)
                
                ### Lv0 figures
                level = 0
                save_flows(flows_Lv0, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv0_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                save_flows(pred_flows_Lv0_-flows_Lv0.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)

                ### Lv1 figures
                level = 1
                save_flows(flows_Lv1, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv1_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                save_flows(pred_flows_Lv1_-flows_Lv1.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)

                ### Lv2 figures
                level = 2
                save_flows(flows_Lv2, name=name, img_dir = self.sub_img_dir, type_name = 'ref', level = level, epoch=epoch)
                save_flows(pred_flows_Lv2_, name=name, img_dir = self.sub_img_dir, type_name = 'pred', level = level, epoch=epoch)
                save_flows(pred_flows_Lv2_-flows_Lv2.cpu(), name=name, img_dir = self.sub_img_dir, type_name = 'error', level = level, epoch=epoch)
                
                self.timer.stop()
                self.elapsed_times[f'save_figs_{name}'].append(self.timer.elapsed_seconds())

        # Horovod: average metric values across workers.
        losses = {}
        for level in range(3):
            losses[f'log_loss_{name}_{self.loss_type}_Lv{level}'] = log_loss[level]

        for key, value in losses.items():
            loss = super()._metric_average(value, key)
            self.loss_dict[key].append(loss)

    ### For inference
    def _infer(self):
        with torch.no_grad():
            self._convert(data_loader=self.val_loader, name='validation')
            self._convert(data_loader=self.test_loader, name='test')

    def _convert(self, data_loader, name):
        self.model0.eval()
        self.model1.eval()
        self.model2.eval()
        nb_samples = len(data_loader.sampler)
        
        for indices, sdf, flows in data_loader:
            # Load data and meta-data
            sdf_Lv0, sdf_Lv1, sdf_Lv2 = sdf
            flows_Lv0, flows_Lv1, flows_Lv2 = flows
            
            _, patch_y_Lv1, patch_x_Lv1, *_ = sdf_Lv1.shape
            _, patch_y_Lv2, patch_x_Lv2, *_ = sdf_Lv2.shape
            
            # Number of sub patches in each level
            nb_patches_Lv2 = patch_y_Lv2 * patch_x_Lv2
            nb_patches_Lv1 = patch_y_Lv1 * patch_x_Lv1
            
            # Sub patch inside the Lv1 patch
            patch_y_Lv2 = patch_y_Lv2 // patch_y_Lv1
            patch_x_Lv2 = patch_x_Lv2 // patch_x_Lv1
            
            batch_len = len(sdf_Lv0)
            
            ## To device
            self.timer.start()
            
            sdf_Lv0, sdf_Lv1, sdf_Lv2 = sdf_Lv0.to(self.device), sdf_Lv1.to(self.device), sdf_Lv2.to(self.device)
            flows_Lv0, flows_Lv1, flows_Lv2 = flows_Lv0.to(self.device), flows_Lv1.to(self.device), flows_Lv2.to(self.device)
            
            self.timer.stop()
            self.elapsed_times[f'MemcpyH2D_{name}'].append(self.timer.elapsed_seconds())
            
            # Keep sdfs on CPUs
            sdf_Lv0_cpu = sdf_Lv0.to('cpu')
            sdf_Lv1_cpu = sdf_Lv1.to('cpu')
            sdf_Lv2_cpu = sdf_Lv2.to('cpu')
            
            ## Normalization or standardization
            sdf_Lv0 = super()._preprocess(sdf_Lv0, self.sdf_Lv0_var0, self.sdf_Lv0_var1)
            sdf_Lv1 = super()._preprocess(sdf_Lv1, self.sdf_Lv1_var0, self.sdf_Lv1_var1)
            sdf_Lv2 = super()._preprocess(sdf_Lv2, self.sdf_Lv2_var0, self.sdf_Lv2_var1)
            
            flows_Lv0 = super()._preprocess(flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            flows_Lv1 = super()._preprocess(flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
            flows_Lv2 = super()._preprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
            
            # Objectives: construct pred_flows_Lv0-Lv2
            pred_flows_Lv0_ = torch.zeros_like(flows_Lv0, device='cpu')
            pred_flows_Lv1_ = torch.zeros_like(flows_Lv1, device='cpu')
            pred_flows_Lv2_ = torch.zeros_like(flows_Lv2, device='cpu')
            
            #### Train Lv0
            self.timer.start()
            sdf_Lv0_ = sdf_Lv0[:, 0, 0]
            flows_Lv0_ = flows_Lv0[:, 0, 0]

            ### Update weights
            pred_flows_Lv0 = self.model0(sdf_Lv0_)
            
            level = 0
            
            ### Destandardization and save
            pred_flows_Lv0 = super()._postprocess(pred_flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            pred_flows_Lv0_[:, 0, 0, :, :, :] = pred_flows_Lv0.detach().cpu()
            
            self.timer.stop()
            self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())
            
            ### Train Lv1
            for iy_Lv1, ix_Lv1 in itertools.product(range(patch_y_Lv1), range(patch_x_Lv1)):
                self.timer.start()
                sdf_Lv1_ = sdf_Lv1[:, iy_Lv1, ix_Lv1]
                flows_Lv1_ = flows_Lv1[:, iy_Lv1, ix_Lv1]
                
                ### Update weights
                pred_flows_Lv1 = self.model1(sdf_Lv1_)
                
                ### Log losses
                level = 1
                
                pred_flows_Lv1 = super()._postprocess(pred_flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
                pred_flows_Lv1_[:, iy_Lv1, ix_Lv1, :, :, :] = pred_flows_Lv1.detach().cpu()
                
                self.timer.stop()
                self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())
                
                ### Train Lv2
                for iy_Lv2, ix_Lv2 in itertools.product(range(patch_y_Lv2), range(patch_x_Lv2)):
                    self.timer.start()
                    global_ix_Lv2 = ix_Lv2 + (ix_Lv1 * patch_x_Lv2)
                    global_iy_Lv2 = iy_Lv2 + (iy_Lv1 * patch_y_Lv2)
                    sdf_Lv2_   = sdf_Lv2[:, global_iy_Lv2, global_ix_Lv2]
                    flows_Lv2_ = flows_Lv2[:, global_iy_Lv2, global_ix_Lv2]
                
                    ### Update generator weights
                    pred_flows_Lv2 = self.model2(sdf_Lv2_)
                
                    level = 2
                
                    ### Destandardization and save
                    pred_flows_Lv2 = super()._postprocess(pred_flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
                    pred_flows_Lv2_[:, global_iy_Lv2, global_ix_Lv2, :, :, :] = pred_flows_Lv2.detach().cpu()
                
                    self.timer.stop()
                    self.elapsed_times[f'{name}_Lv{level}'].append(self.timer.elapsed_seconds())

            # Save the data in netcdf format
            self.timer.start()
            flows_Lv0 = super()._postprocess(flows_Lv0, self.flows_Lv0_var0, self.flows_Lv0_var1)
            flows_Lv1 = super()._postprocess(flows_Lv1, self.flows_Lv1_var0, self.flows_Lv1_var1)
            flows_Lv2 = super()._postprocess(flows_Lv2, self.flows_Lv2_var0, self.flows_Lv2_var1)
            
            ### Zeros inside objects
            pred_flows_Lv0_ = super()._zeros_inside_objects(pred_flows_Lv0_, sdf_Lv0_cpu)
            pred_flows_Lv1_ = super()._zeros_inside_objects(pred_flows_Lv1_, sdf_Lv1_cpu)
            pred_flows_Lv2_ = super()._zeros_inside_objects(pred_flows_Lv2_, sdf_Lv2_cpu)
            
            ### Lv0 figures
            level = 0
            save_as_netcdf(sdf=sdf_Lv0_cpu, real_flows=flows_Lv0.cpu(), pred_flows=pred_flows_Lv0_,
                           indices=indices, epoch=self.epoch_start, level=level, name=name, data_dir=self.inference_dir)
            
            ### Lv1 figures
            level = 1
            save_as_netcdf(sdf=sdf_Lv1_cpu, real_flows=flows_Lv1.cpu(), pred_flows=pred_flows_Lv1_,
                           indices=indices, epoch=self.epoch_start, level=level, name=name, data_dir=self.inference_dir)
            
            ### Lv2 figures
            level = 2
            save_as_netcdf(sdf=sdf_Lv2_cpu, real_flows=flows_Lv2.cpu(), pred_flows=pred_flows_Lv2_,
                           indices=indices, epoch=self.epoch_start, level=level, name=name, data_dir=self.inference_dir)
            
            self.timer.stop()
            self.elapsed_times[f'save_figs_{name}'].append(self.timer.elapsed_seconds())
