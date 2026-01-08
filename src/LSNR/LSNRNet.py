"""
Local SNR Network (LSNRNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LSNR.modules import *
from mpSE._utils import *

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
    "encoder" : {
        "enc1" : {
            "in_channels": 1, 
            "out_channels": 64,
            "kernel_size": [1, 5],
            "stride": [1,2],
            "padding": [0,0]
        },
        "enc2" : {
            "in_channels": 64, 
            "out_channels": 128,
            "kernel_size": [1, 5],
            "stride": [1,2],
            "padding": [0,0]
        },
        "enc3" : {
            "in_channels": 128, 
            "out_channels": 256,
            "kernel_size": [1, 5],
            "stride": [1,2],
            "padding": [0,0]
        }
    },
    "decoder" : {
        "dec1" : {
            "in_channels": 512, 
            "out_channels": 128,
            "kernel_size": [1, 5],
            "stride": [1,2],
            "padding": [0,0],
            "output_padding" :[0,1]
        },
        "dec2" : {
            "in_channels": 256, 
            "out_channels": 64,
            "kernel_size": [1,5],
            "stride": [1,2],
            "padding": [0,0],
        },
        "dec3" : {
            "in_channels": 128, 
            "out_channels": 1,
            "kernel_size": [1, 5],
            "stride": [1,2],
            "padding": [0,0],
            "output_padding" :[0,0]
        }
    },
    "temporal" : {
        "hidden_size" : 256
    },
    "frequential" : {
        "hidden_size" : 256
    },
    "mask":{
        "type" : "sigmoid",
        "hidden_size" : 257
    },
    "LSNR" : {
        "location" : "None", # bottleneck1, bottleneck2, bottleneck3
        "hidden_size" : 7424 # 29 * 256
    }
}

class LSNRNet(nn.Module) : 
    def __init__(self,architecture=default_arch) : 
        super(LSNRNet,self).__init__()

        # Module class
        spectrum = getattr(import_module("mpSE._edges"), "MagEncoder")
        encoder = getattr(import_module("mpSE._coders"), "SimpleEncoder")
        decoder = getattr(import_module("mpSE._coders"), "SimpleDecoder")
        frequential = getattr(import_module("mpSE._necks"), "SimpleFBlock")
        temporal = getattr(import_module("mpSE._necks"), "SimpleTBlock")

        masking = globals()[architecture["mask"]["type"]]

        # Module Def
        self.enc = []
        self.dec = []
        self.n_coder = len(architecture["encoder"])

        self.spectrum = spectrum(**architecture["spectrum"])

        for i in range(self.n_coder) : 
            module = encoder(**architecture["encoder"][f"enc{i+1}"])
            self.add_module(f"enc{i+1}",module)
            self.enc.append(module)

        for i in range(self.n_coder) : 
            module = decoder(**architecture["decoder"][f"dec{i+1}"])
            self.add_module(f"dec{i+1}",module)
            self.dec.append(module)

        self.f_block = frequential(**architecture["frequential"])
        self.t_block = temporal(**architecture["temporal"])

        self.masking = masking(**architecture["mask"],lsnr_location=architecture["LSNR"]["location"])

        self.lsnr_layer = getattr(import_module("LSNR.modules"),architecture["LSNR"]["type"])(**architecture["LSNR"])
        self.lsrn_location = architecture["LSNR"]["location"]
        print(f"LSNRNet::lsnr_location : {self.lsrn_location}")

        self.act = nn.Sigmoid()

        # Module apply
        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)

    def forward(self, x) :
        # x : B,C,T,F
        B, C, T, F = x.shape
        x_in = x.clone()

        x = self.spectrum(x)  # B,2,T,F  -> B,C,T,

        skip = []
        for i in range(self.n_coder) : 
            x = self.enc[i](x)
            skip.append(x)
            printSimple(f"LSNRNet::enc[{i}] : {x.shape}")
            #print(x.shape)
        lsnr = None
        if self.lsrn_location == "bottleneck1" :
            lsnr = self.lsnr_layer(x)
        x = self.f_block(x)
        if self.lsrn_location == "bottleneck2" :
            lsnr = self.lsnr_layer(x)
        x = self.t_block(x)
        printSimple("bottleneck3 : ",x.shape)
        if self.lsrn_location == "bottleneck3" :
            lsnr = self.lsnr_layer(x)

        printSimple(f"LSNRNet::bottleneck : {x.shape}")

        for i in range(self.n_coder) : 
            x = self.dec[i](torch.cat((x,skip[self.n_coder - i - 1]),dim=1))
            printSimple(f"LSNRNet::dec[{i}] : {x.shape}")
            #print(x.shape)

        y = self.masking(x_in, x,lsnr)

        if lsnr is None :
            lsnr = torch.zeros((B,T,1), device=x.device)
        return  y, lsnr

class LSNRNetHelper(nn.Module) : 
    def __init__(self,n_fft=512,n_hop=256, architecture = default_arch ,**kwargs) : 
        super(LSNRNetHelper,self).__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop

        self.window = get_window(n_fft = n_fft, n_hop=n_hop)
        self.model = LSNRNet(architecture=architecture)

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

