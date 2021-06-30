import torch
import torch.multiprocessing as mp
import pathlib
import horovod.torch as hvd
import numpy as np
import itertools
import time
from collections import defaultdict
from torch.utils.data import DataLoader
from .flow_dataset import FlowDataset

class Timer:
    def __init__(self, device='cuda'):
        self.device = device
        if self.device == 'cuda':
            self.start_ = torch.cuda.Event(enable_timing=True)
            self.end_   = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.device == 'cuda':
            self.start_.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.device == 'cuda':
            self.end_.record()
            torch.cuda.synchronize()
            self.elapsed_ms_ = self.start_.elapsed_time(self.end_)
        else:
            self.elapsed_ms_ = (time.time() - self.start_time) * 1.e3

    def elapsed_ms(self):
        return self.elapsed_ms_

    def elapsed_seconds(self):
        return self.elapsed_ms_ * 1.e-3

class MeasureMemory:
    def __init__(self, device = 'cuda'):
        self.reserved_ = 0.
        self.alloc_ = 0.
        self.device = device

    def measure(self):
        if 'cuda' in self.device:
            self.reserved_ = torch.cuda.memory_reserved(device=self.device) / 1.e9
            self.alloc_ = torch.cuda.memory_allocated(device=self.device) / 1.e9

    def reserved(self):
        return self.reserved_

    def alloc(self):
        return self.alloc_

class _BaseTrainer:
    """
    Base class for training
    """

    def __init__(self, **kwargs):
        self.loss_dict = defaultdict(list)
        self.elapsed_times = defaultdict(list)
        self.memory_consumption = {}

        allowed_kwargs = {
                          'dim',
                          'preprocess_type',
                          'device',
                          'n_epochs',
                          'model_name',
                          'seed',
                          'data_dir',
                          'batch_size',
                          'run_number',
                          'validation_ratio',
                          'test_ratio',
                          'lr',
                          'beta_1',
                          'beta_2',
                          'use_adasum',
                          'fp16_allreduce',
                          'gradient_predivide_factor',
                          'loss_type',
                          'padding_mode',
                         }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        data_dir = kwargs.get('data_dir')
        if not data_dir:
            raise ValueError('Argument data_dir must be given')
        self.data_dir = data_dir
        
        self.dim = kwargs.get('dim', 2)
        self.preprocess_type = kwargs.get('preprocess_type', 'normalization')
        self.padding_mode = kwargs.get('padding_mode', 'reflect')
        self.device = kwargs.get('device', 'cuda')
        self.seed   = kwargs.get('seed', 0)
        self.batch_size = kwargs.get('batch_size', 16)
        self.run_number = kwargs.get('run_number', 0)
        self.validation_ratio = kwargs.get('validation_ratio', 0.025)
        self.test_ratio = kwargs.get('test_ratio', 0.025)
        self.lr         = kwargs.get('lr', 0.0002)
        self.beta_1     = kwargs.get('beta_1', 0.9)
        self.beta_2     = kwargs.get('beta_2', 0.999)
        self.dropout    = kwargs.get('dropout', 0.)
        self.use_adasum = kwargs.get('use_adasum', False)
        self.fp16_allreduce = kwargs.get('fp16_allreduce', False)
        self.gradient_predivide_factor = kwargs.get('gradient_predivide_factor', 1.0)
        self.loss_type  = kwargs.get('loss_type', 'mae_loss')
        self.scale  = kwargs.get('scale', 2)
        self.n_epochs = kwargs.get('n_epochs', 16)

        self.timer = Timer(device=self.device)

    def initialize(self, **kwargs):
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        raise NotImplementedError()

    def step(self, epoch):
        self._step(epoch)

    def _step(self, epoch):
        raise NotImplementedError()

    def finalize(self, seconds):
        self._finalize(seconds)

    def _finalize(self, seconds):
        raise NotImplementedError()

    def _prepare_dirs(self):
        self.out_dir = pathlib.Path(f'torch_model_MPI{self.size}') / f'{self.model_name}'
        self.img_dir = pathlib.Path('GeneratedImages')
        if self.master:
            if not self.out_dir.exists():
                self.out_dir.mkdir(parents=True)
          
        if not self.img_dir.exists():
            self.img_dir.mkdir(parents=True)

        # Barrier
        hvd.allreduce(torch.tensor(1), name="Barrier")
        
        # Create sub directories
        sub_img_dir = self.img_dir / f'rank{self.rank}'
        if not sub_img_dir.exists():
            sub_img_dir.mkdir(parents=True)

        self.sub_img_dir = sub_img_dir
         
        self.model_dir = self.out_dir / f'rank{self.rank}'
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
                
        levels = np.arange(3)
        modes = ['train', 'test', 'validation']
        for mode, level in itertools.product(modes, levels):
            sub_dir = sub_img_dir / f'{mode}_Lv{level}'
            if not sub_dir.exists():
                sub_dir.mkdir(parents=True)


    def __split_files(self):
        at_least_one = lambda number: 1 if number < 1 else number

        all_files = sorted( list(pathlib.Path(self.data_dir).glob('shot*.h5')) )
        n_samples = len(all_files)
        m_test = int(n_samples * self.test_ratio)
        m_val = int(n_samples * self.validation_ratio)
        m_test = at_least_one(m_test) # [0:m_test] -> test dataset
        m_val  = at_least_one(m_val)  # [m_test:m_test + m_val] -> val dataset
        m_test_and_val = m_test + m_val
        m_train = n_samples - m_test - m_val  # [m_test + m_val:n_samples] -> train dataset
        
        train_files = all_files[m_test_and_val:]
        val_files   = all_files[m_test:m_test_and_val]
        test_files  = all_files[:m_test]
        assert (len(train_files) + len(val_files) + len(test_files)) == n_samples
        
        return train_files, val_files, test_files

    def _dataloaders(self):
        train_files, val_files, test_files = self.__split_files()
        train_dataset = FlowDataset(files=train_files, model_name=self.model_name)
        val_dataset   = FlowDataset(files=val_files, model_name=self.model_name)
        test_dataset  = FlowDataset(files=test_files, model_name=self.model_name)
        
        # Horovod: use DistributedSampler to partition the training data
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                                train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                                val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                                test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.device == 'cuda' else {}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork'
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, **kwargs)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, sampler=self.val_sampler, **kwargs)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, sampler=self.test_sampler, **kwargs)
        
        return train_loader, val_loader, test_loader

    def _metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def _get_attrs(self):
        attrs = {}
        attrs['rank'] = self.rank
        attrs['size'] = self.size
        attrs['device'] = self.device
        attrs['loss_type'] = self.loss_type
        attrs['epoch_start'] = self.epoch_start
        attrs['epoch_end'] = self.epoch_start + self.n_epochs - 1
        attrs['scale'] = self.scale
        attrs['lr'] = self.lr
        attrs['beta_1'] = self.beta_1
        attrs['beta_2'] = self.beta_2
        attrs['dropout'] = self.dropout
        attrs['preprocess_type'] = self.preprocess_type

        return attrs

    def _set_normalization_coefs(self, shape):
        if self.preprocess_type == 'normalization':
            sdf_Lv0_var0, sdf_Lv1_var0, sdf_Lv2_var0 = 3.0999, 3.1055, 3.1082
            sdf_Lv0_var1, sdf_Lv1_var1, sdf_Lv2_var1 = -0.3051, -0.3072, -0.3092
                    
            flows_Lv0_var0, flows_Lv1_var0, flows_Lv2_var0 = [1.2862, 0.5025], [1.2862, 0.5025], [1.2862, 0.5025]
            flows_Lv0_var1, flows_Lv1_var1, flows_Lv2_var1 = [-0.0269, -0.4921], [-0.1085, -0.4922], [-0.2665, -0.4922]
        
        elif self.preprocess_type == 'standardization':
            sdf_Lv0_var0, sdf_Lv1_var0, sdf_Lv2_var0 = 1.3099, 1.3099, 1.3099
            sdf_Lv0_var1, sdf_Lv1_var1, sdf_Lv2_var1 = 0.5772, 0.5772, 0.5772
            
            flows_Lv0_var0, flows_Lv1_var0, flows_Lv2_var0 = [0.8940, 0.0036], [0.8940, 0.0036], [0.8940, 0.0036]
            flows_Lv0_var1, flows_Lv1_var1, flows_Lv2_var1 = [0.2452, 0.1005], [0.2452, 0.1005], [0.2452, 0.1005]
        
        ## Conver to tensors
        to_tensor = lambda var: torch.tensor(var).view(*shape).float().to(self.device)
        
        self.sdf_Lv0_var0, self.sdf_Lv1_var0, self.sdf_Lv2_var0 = to_tensor(sdf_Lv0_var0), to_tensor(sdf_Lv1_var0), to_tensor(sdf_Lv2_var0)
        self.sdf_Lv0_var1, self.sdf_Lv1_var1, self.sdf_Lv2_var1 = to_tensor(sdf_Lv0_var1), to_tensor(sdf_Lv1_var1), to_tensor(sdf_Lv2_var1)
         
        self.flows_Lv0_var0, self.flows_Lv1_var0, self.flows_Lv2_var0 = to_tensor(flows_Lv0_var0), to_tensor(flows_Lv1_var0), to_tensor(flows_Lv2_var0)
        self.flows_Lv0_var1, self.flows_Lv1_var1, self.flows_Lv2_var1 = to_tensor(flows_Lv0_var1), to_tensor(flows_Lv1_var1), to_tensor(flows_Lv2_var1)

    def __normalize(self, x, *args):
        """
        Data range to be [0, 1] or [-1, 1]
        """

        xmax, xmin, scale = args

        x -= xmin
        x /= (xmax - xmin)

        if not scale == 1:
            x = scale * (x - 0.5)

        return x

    def __denormalize(self, x, *args):
        """
        Data range to be [0, 1] or [-1, 1]
        """

        xmax, xmin, scale = args
        if not scale == 1:
            x = x / scale + 0.5

        x = x * (xmax - xmin) + xmin
        return x

    def __standardize(self, x, *args):
        """
        mean to 0 and std to 1
        """
        mean, std = args
        x -= mean
        x /= std
        return x

    def __destandardize(self, x, *args):
        mean, std = args
        x = (x * std) + mean
        return x

    def _preprocess(self, x, *args):
        """
        Standardize the mean of data to be 0 and std to be 1
        """

        if self.preprocess_type == 'normalization':
            return self.__normalize(x, *args, self.scale)
        elif self.preprocess_type == 'standardization':
            return self.__standardize(x, *args)
        else:
            return x

    def _postprocess(self, x, *args):
        if self.preprocess_type == 'normalization':
            return self.__denormalize(x, *args, self.scale)
        elif self.preprocess_type == 'standardization':
            return self.__destandardize(x, *args)
        else:
            return x

    def _zeros_inside_objects(self, flows, SDF):
        return torch.where(SDF <= 0, 0., flows.double()).float()
