"""
Model from https://github.com/NVIDIA/pix2pixHD
"""

import torch
import numpy as np
from torch import nn
from .blocks import ConvBlock, DeconvBlock, ResnetBlock, PoolBlock

### Define a global generator
class GlobalGenerator(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        allowed_kwargs = {
                          'in_channels',
                          'out_channels',
                          'hidden_dim',
                          'dim',
                          'n_downsampling',
                          'n_blocks',
                          'padding_mode',
                          'dropout',
                         }
         
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        in_channels    = kwargs.get('in_channels', 1)
        out_channels   = kwargs.get('out_channels', 2)
        hidden_dim     = kwargs.get('hidden_dim', 64)
        dim            = kwargs.get('dim', 2)
        n_downsampling = kwargs.get('n_downsampling', 3)
        n_blocks       = kwargs.get('n_blocks', 9)
        dropout        = kwargs.get('dropout', 0.)
        
        padding_mode = kwargs.get('padding_mode', 'zeros')
        layers = []

        layers += [ConvBlock(in_channels, hidden_dim, kernel_size=7, stride=1, dim=dim, padding_mode=padding_mode)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            layers += [ConvBlock(hidden_dim * mult, hidden_dim * mult * 2, kernel_size=3, stride=2, dim=dim, padding_mode=padding_mode)]

        ### Resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            # Do not use dropout for global generator
            layers += [ResnetBlock(in_channels=hidden_dim * mult, dim=dim, padding_mode=padding_mode, dropout=0.)]

        ### upsample, the output of this layer is fed to local enhancer
        ### Instead of transposeConv2D, we use upsampling to avoid checkerboard problems
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            layers += [DeconvBlock(hidden_dim * mult, int(hidden_dim * mult / 2), kernel_size=3, stride=2, dim=dim)]

        # 1x1 by convolution to generate flows
        layers += [ConvBlock(hidden_dim, out_channels, kernel_size=7, stride=1, dim=dim, padding_mode=padding_mode, activation='Tanh')]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

### Define a local enhancer (pix2pixHD)
class LocalEnhancer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        allowed_kwargs = {
                          'in_channels',
                          'out_channels',
                          'hidden_dim',
                          'dim',
                          'n_downsampling',
                          'n_blocks_global',
                          'n_blocks_local',
                          'n_local_enhancers',
                          'padding_mode',
                          'patched',
                          'dropout',
                         }
        
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        in_channels = kwargs.get('in_channels', 1)
        out_channels = kwargs.get('out_channels', 2)
        hidden_dim = kwargs.get('hidden_dim', 16)
        dim = kwargs.get('dim', 2)
        n_blocks_local = kwargs.get('n_blocks_local', 3)
        self.n_local_enhancers = kwargs.get('n_local_enhancers', 2)
        
        padding_mode = kwargs.get('padding_mode', 'zeros')
        dropout = kwargs.get('dropout', 0.)
        self.patched = kwargs.get('patched', False)
        
        # Fed to global generator
        kwargs_global = kwargs.copy()
        local_keys = {'n_blocks_local', 'patched'}
        for key in local_keys:
            kwargs_global.pop(key, None)

        # The hidden_dim in global generator must be hidden_dim * (2**n_local_enhancers)
        kwargs_global['hidden_dim'] = hidden_dim * (2**self.n_local_enhancers)
        
        ###### Global generator model ######
        model_global = GlobalGenerator(**kwargs_global)
        nb_layers = len(model_global.model)
        
        nb_front_layers = nb_layers - 1
        
        model_global_coarse = [model_global.model[i] for i in range(nb_front_layers)] # get rid of final convolutional layers
        model_global_upsample = [model_global.model[i] for i in range(nb_front_layers, nb_layers)]
        self.model_coarse = nn.Sequential(*model_global_coarse)
        self.model_coarse_final = nn.Sequential(*model_global_upsample)

        ###### Local enhancer model #######
        for n in range(1, self.n_local_enhancers+1):
            ### downsample
            hidden_dim_tmp = hidden_dim * (2**(self.n_local_enhancers - n))
            downsample_layers = []
            downsample_layers += [ConvBlock(in_channels, hidden_dim_tmp, kernel_size=7, stride=1, padding_mode=padding_mode)]
            downsample_layers += [ConvBlock(hidden_dim_tmp, hidden_dim_tmp * 2, kernel_size=3, stride=2, padding_mode=padding_mode)]
            setattr(self, f'model_downsample_{n}', nn.Sequential(*downsample_layers))
            
            ### Resnet blocks
            upsample_layers = []
            for i in range(n_blocks_local):
                upsample_layers += [ResnetBlock(in_channels=hidden_dim_tmp * 2, dim=dim, padding_mode=padding_mode, dropout=dropout)]
                
            ### upsample
            upsample_layers += [DeconvBlock(hidden_dim_tmp * 2, hidden_dim_tmp, kernel_size=3, stride=2)]
            setattr(self, f'model_upsample_{n}', nn.Sequential(*upsample_layers))
            
            ### final convolution
            final_layers = [ConvBlock(hidden_dim_tmp, out_channels, kernel_size=7, stride=1, padding_mode=padding_mode, activation='Tanh')]
            setattr(self, f'model_final_{n}', nn.Sequential(*final_layers))
                 
        self.downsample = PoolBlock(kernel_size=3, stride=2, padding=1, dim=dim)

    def trainable(self, level, requires_grad):
        if level == 0:
            # Make global generator part trainable
            layers = [self.model_coarse, self.model_coarse_final]
        else:
            layers = []
            layers += [getattr(self, f'model_downsample_{level}')]
            layers += [getattr(self, f'model_upsample_{level}')]
            layers += [getattr(self, f'model_final_{level}')]
             
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(self, sdf, patch_ranges=None):
        if self.patched:
            return self.forward_patched_local_enhancer(sdf, patch_ranges)
        else:
            return self.forward_local_enhancer(sdf)

    def forward_local_enhancer(self, sdfs):
        # Pix2PixHD
        ### Global generator output for entire image
        out_low = self.model_coarse(sdfs[0])
        
        ### Coarse to fine
        for n in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, f'model_downsample_{n}')
            model_upsample   = getattr(self, f'model_upsample_{n}')
            model_final      = getattr(self, f'model_final_{n}')
            out_low          = model_upsample(model_downsample(sdfs[n]) + out_low)
            flows            = model_final(out_low)
                
        return flows

    def forward_patched_local_enhancer(self, sdfs, patch_ranges=None):
        if patch_ranges is None:
            return self._forward_global(sdfs)
        else:
            return self._forward_local(sdfs, patch_ranges)

    def _forward_global(self, sdfs):
        out_low = self.model_coarse(sdfs)
        global_low = self.model_coarse_final(out_low)
        return global_low
         
    def _forward_local(self, sdfs, patch_ranges):
        ### Encoding with Global generator
        out_low = self.model_coarse(sdfs[0])
    
        ### coarse to fine
        n_refine = len(patch_ranges)
        for n in range(1, n_refine+1):
            model_downsample = getattr(self, f'model_downsample_{n}')
            model_upsample = getattr(self, f'model_upsample_{n}')
            model_final = getattr(self, f'model_final_{n}')
            patch_range = patch_ranges[n-1]
            out_low = model_upsample(model_downsample(sdfs[n]) + out_low[patch_range])
            flows = model_final(out_low)
    
        return flows
