import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpSE._utils import *

from importlib import import_module

default_arch = {
    "encoder" : {
        "enc1" : {
            "in_channels": 2, 
            "out_channels": 64,
            "kernel_size": [5, 1],
            "stride": [2,1],
            "padding": [0,0]
        },
        "enc2" : {
            "in_channels": 64, 
            "out_channels": 128,
            "kernel_size": [5, 1],
            "stride": [2,1],
            "padding": [0,0]
        }
    },
    "decoder" : {
        "dec1" : {
            "in_channels": 256, 
            "out_channels": 64,
            "kernel_size": [5, 1],
            "stride": [2,1],
            "padding": [0,0]
        },
        "dec2" : {
            "in_channels": 128, 
            "out_channels": 2,
            "kernel_size": [5, 1],
            "stride": [2,1],
            "padding": [0,0]
        }
    },
    "temporal" : {
        "hidden_size" : 128
    },
    "frequential" : {
        "hidden_size" : 128
    }
}

class SimpleNet(nn.Module) : 
    def __init__(self,architecture=default_arch) : 
        super(SimpleNet,self).__init__()

        # Module class
        encoder = getattr(import_module("mpSE._coders"), "SimpleEncoder")
        decoder = getattr(import_module("mpSE._coders"), "SimpleDecoder")
        frequential = getattr(import_module("mpSE._necks"), "SimpleFBlock")
        temporal = getattr(import_module("mpSE._necks"), "SimpleTBlock")

        # Module Def
        self.enc = []
        self.dec = []
        self.n_coder = len(architecture["encoder"])

        for i in range(self.n_coder) : 
            module = encoder(**architecture["encoder"][f"enc{i+1}"])
            self.add_module(f"enc{i+1}",module)
            self.enc.append(module)

        for i in range(self.n_coder) : 
            module = decoder(**architecture["decoder"][f"dec{i+1}"])
            self.add_module(f"dec{i+1}",module)
            self.dec.append(module)

        self.f_block = frequential(**architecture["temporal"])
        self.t_block = temporal(**architecture["temporal"])

        self.act = nn.Sigmoid()

        # Module apply
        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)

    def forward(self, x) :
        # x : B,F,T,2 -> B,2,F,T
        x = x.permute(0,3,1,2)

        skip = []
        x_orig = x.clone()
        for i in range(self.n_coder) : 
            x = self.enc[i](x)
            skip.append(x)
            #print(x.shape)

        x = self.f_block(x)
        x = self.t_block(x)

        for i in range(self.n_coder) : 
            x = self.dec[i](torch.cat((x,skip[self.n_coder - i - 1]),dim=1))
            #print(x.shape)

        mask = self.act(x)
        y = mask*x_orig    

        # y : B,2,F,T -> B,F,T,2
        y = y.permute(0,2,3,1)

        return  y

class SimpleNetHelper(nn.Module) : 
    def __init__(self,n_fft=256,n_hop=128,**kwargs) : 
        super(SimpleNetHelper,self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop

        self.window = get_window(n_fft = n_fft, n_hop=n_hop)

        self.m = SimpleNet()

    def forward(self,x) :
        X = self.analysis(x)
        Y = self.m(X)
        y = self.synthesis(Y)

        return y

    def analysis(self,x) : 
        X = torch.stft(x, n_fft = self.n_fft, hop_length = self.n_hop, window = self.window.to(x.device),return_complex = False)
        return X

    def synthesis(self, X) : 
        # X : [B,F,T,2]
        X = X[...,0] + X[...,1] * 1j
        x = torch.istft(X, self.n_fft, self.n_hop, self.n_fft, self.window.to(X.device))
        return x
