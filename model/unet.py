import torch
import numpy as np
from torch import nn
from .blocks import ConvBlock, DeconvBlock

class UNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        allowed_kwargs = {
                          'in_channels',
                          'out_channels',
                          'hidden_dim',
                          'dim',
                          'padding_mode',
                          'n_layers',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        in_channels    = kwargs.get('in_channels', 1)
        out_channels   = kwargs.get('out_channels', 2)
        hidden_dim     = kwargs.get('hidden_dim', 64)
        dim            = kwargs.get('dim', 2)
        padding_mode   = kwargs.get('padding_mode', 'zeros')
        self.n_layers  = kwargs.get('n_layers', 6)

        self.conv0   = ConvBlock(in_channels, hidden_dim, kernel_size=7, stride=1, dim=dim, padding_mode=padding_mode)
        self.deconv0 = ConvBlock(hidden_dim*2, out_channels, kernel_size=7, stride=1, dim=dim, padding_mode='zeros', activation='Tanh')

        for i in range(1, self.n_layers):
            # Encoder layers
            mult = hidden_dim * 2**(i-1)
            setattr(self, f'conv{i}', ConvBlock(mult, mult*2, kernel_size=3, stride=2, dim=dim, padding_mode=padding_mode))

            # Decoder layers
            final_layer = (i == self.n_layers-1)
            in_channels_ = mult*2 if final_layer else mult*4
            setattr(self, f'deconv{i}', DeconvBlock(in_channels_, mult, kernel_size=3, stride=2, dim=dim))

    def forward(self, x):
        ### Encoding
        encoded = []
        out = x
        for i in range(self.n_layers):
            model_encode = getattr(self, f'conv{i}')
            out = model_encode(out)
            encoded.append(out)

        ### Decoding
        for i in reversed(range(self.n_layers)):
            model_decode = getattr(self, f'deconv{i}')
            if i == self.n_layers-1:
                out = model_decode(out)
            else:
                out = model_decode(torch.cat([out, encoded[i]], dim=1))

        return out
