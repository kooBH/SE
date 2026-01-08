"""
Local SNR Network (LSNRNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpSE._utils import *
from LSNR.modules import lsnr_estimator, SNRAwareMagMaskingV1, SNRAwareMagMaskingV2

from importlib import import_module

print_ENABLED = False
def printSimple(*args, **kwargs):
    if print_ENABLED:
        print(*args, **kwargs)

default_arch = {
    "spectrum": {
        "type" : "magnitude",
        "LPC" : True
    },
}

class Transformer(nn.Module):
    def __init__(self,hidden_size=512, num_heads=8, ff_excitation=4):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_excitation),
            nn.ReLU(),
            nn.Linear(hidden_size * ff_excitation, hidden_size)
        )
        self.norm_ff = nn.LayerNorm(hidden_size)

    def forward(self,x) : 
        # B,T,D
        x = self.attn(x, x, x)[0]  
        x = self.norm_attn(x + x)  
        x = self.ff(x)
        x = self.norm_ff(x + x)

        return x

class TFBlock(nn.Module) : 
    def __init__(self, dim_time = 30,dim_channel=64, num_heads=8, ff_excitation=4, n_repeats=2):
        super(TFBlock, self).__init__()

        self.f_block = Transformer(hidden_size=dim_channel, num_heads=num_heads, ff_excitation=ff_excitation)
        self.t_block = Transformer(hidden_size=dim_time, num_heads=num_heads, ff_excitation=ff_excitation)

    def forward(self,x):
        # x :  B,C,T,F
        B, C, T, F = x.shape

        # B,C,T,F -> B*T,F,C
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * T, F, C) 
        x = self.f_block(x)

        # B*T,F,C ->B*C,T,F
        x = x.reshape(B,T,F,C)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B * C, T, F)
        x = self.t_block(x)

        # B*C,T,F -> B,C,T,F
        x = x.reshape(B, C, T, F)

        return x


class LSNRFormer(nn.Module) : 
    def __init__(self,architecture=default_arch) : 
        super(LSNRFormer,self).__init__()

        # Module class
        spectrum = getattr(import_module("mpSE._edges"), "MagEncoder")

        masking = globals()[architecture["mask"]["type"]]

        # Module Def
        self.enc = []
        self.dec = []
        self.n_coder = len(architecture["encoder"])

        self.spectrum = spectrum(**architecture["spectrum"])

    def forward(self, x) :
        # x : B,C,T,F
        B, C, T, F = x.shape
        x_in = x.clone()

        x = self.spectrum(x)  # B,2,T,F  -> B,C,T,

        y = self.masking(x_in, x,lsnr)

        if lsnr is None :
            lsnr = torch.zeros((B,T,1), device=x.device)
        return  y, lsnr

class LSNRFormerHelper(nn.Module) : 
    def __init__(self,n_fft=512,n_hop=256, architecture = default_arch ,**kwargs) : 
        super(LSNRFormerHelper,self).__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop

        self.window = get_window(n_fft = n_fft, n_hop=n_hop)
        self.model = LSNRFormer(architecture=architecture)

    def forward(self,x) :
        # X : B,F,T,2 
        X = self.analysis(x)

        # X : B,F,T,2  -> B,2,T,F
        X = X.permute(0,3,2,1)

        Y,lsnr = self.model(X)

        # B,2,T,F-> B,F,T,2
        Y = Y.permute(0,3,2,1)
        y = self.synthesis(Y)

        return y,lsnr

    def analysis(self,x) : 
        X = torch.stft(x, n_fft = self.n_fft, hop_length = self.n_hop, window = self.window.to(x.device),return_complex = False)
        return X

    def synthesis(self, X) : 
        # X : [B,F,T,2]
        X = X[...,0] + X[...,1] * 1j
        x = torch.istft(X, self.n_fft, self.n_hop, self.n_fft, self.window.to(X.device))
        return x
    
    def to_onnx(self, output_path, device='cpu'):
        print("Not implemented.")

