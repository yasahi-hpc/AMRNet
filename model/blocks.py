"""
Model from https://github.com/NVIDIA/pix2pixHD
"""

import torch
import numpy as np
from torch import nn

# Wrappers to manage 2D or 3D problems
def pad_layer(dim, padding_mode='reflect'):
    if padding_mode == 'reflect':
        return nn.ReflectionPad2d if dim == 2 else nn.ReflectionPad3d
    elif padding_mode == 'replicate':
        return nn.ReplicationPad2d if dim == 2 else nn.ReplicationPad3d
    else:
        raise NotImplementedError(f'padding {padding_mode} is not implemented')

def conv_layer(dim):
    return nn.Conv2d if dim == 2 else nn.Conv3d

def norm_layer(dim):
    return nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

def pool_layer(dim, pool_mode='avg'):
    if pool_mode == 'avg':
        return nn.AvgPool2d if dim == 2 else nn.AvgPool3d
    elif pool_mode == 'max':
        return nn.MaxPool2d if dim == 2 else nn.MaxPool3d
    else:
        raise NotImplementedError(f'pool {pool_mode} is not implemented')

def activation_layer(activation='ReLU'):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'ReakyReLU':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'GELU':
        return nn.GELU()

### Define blocks in the network
class PoolBlock(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, dim=2, pool_mode='avg'):
        super().__init__()

        self.model = pool_layer(dim=dim, pool_mode=pool_mode)(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.model(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dim=2,
                 padding_mode='reflect', activation='ReLU'):
            
        super().__init__()
        layers = []

        assert dim == 2 or dim == 3
         
        p = int(np.ceil((kernel_size-1.0)/2))
        if padding_mode in ['reflect', 'replicate']:
            layers += [pad_layer(dim=dim, padding_mode=padding_mode)(p)]
            p = 0
        elif padding_mode == 'zeros':
            pass
        else:
            raise NotImplementedError(f'padding {padding_mode} is not implemented')
            
        layers += [conv_layer(dim=dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p)]
        layers += [norm_layer(dim=dim)(out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)
                 
    def forward(self, x):
        return self.model(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dim=2,
                 activation='ReLU'):
           
        super().__init__()
        layers = []
        
        p = int(np.ceil((kernel_size-1.0)/2))
        layers += [nn.Upsample(scale_factor=stride)]
        layers += [conv_layer(dim=dim)(in_channels, out_channels, kernel_size=kernel_size, padding=p)]
        layers += [norm_layer(dim=dim)(out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dim=2, padding_mode='reflect', dropout=0.):
        super().__init__()
        layers = []
        layers += [ConvBlock(in_channels, in_channels, kernel_size=3, dim=dim, padding_mode=padding_mode)]
        if dropout > 0.:
            layers += [nn.Dropout(dropout)]
        layers += [ConvBlock(in_channels, in_channels, kernel_size=3, dim=dim,  padding_mode=padding_mode)]
        self.model = nn.Sequential(*layers)
                      
    def forward(self, x):
        out = x + self.model(x) # Skip connection
        return out
