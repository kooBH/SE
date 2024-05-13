import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mpSE.coders import *
from mpSE.necks import *
from mpSE.edges import *
from mpSE.skips import *


architecture_default = {
      "encoder": {
        "enc1": {"in_channels": 1, "out_channels": 64, "kernel_size": [5, 1], "stride": [2,1], "padding": [2,0], "type_norm" : "BatchNorm2d", "type_activation" : "PReLU", "groups": 1 },
        "enc2": {"in_channels": 64, "out_channels": 96, "kernel_size": [5, 1], "stride": [2,1], "padding": [2,0], "type_norm" : "BatchNorm2d", "groups": 64, "type_activation" : "PReLU"},
        "enc3": {"in_channels": 96, "out_channels": 128, "kernel_size": [5, 1], "stride": [2,1], "padding": [2,0], "type_norm" : "BatchNorm2d", "groups": 96, "type_activation" : "PReLU"},
    },
    "decoder": {
        "dec3": {"in_channels": 128, "out_channels": 96, "kernel_size": [5,1], "stride": [2,1], "padding": [2,0], "type_norm" : "BatchNorm2d", "type_activation" : "PReLU", "groups" : 128,
        "output_padding": [1,0]},
        "dec2": {"in_channels": 96, "out_channels": 64, "kernel_size": [5,1], "stride": [2,1], "padding": [2,0], "type_norm" : "BatchNorm2d", "type_activation" : "PReLU", "groups" : 96,"output_padding" : [1,0]},
        "dec1": {"in_channels": 64, "out_channels": 1, "kernel_size": [5,1], "stride": [2,1], "padding": [2,0], "type_norm" : "BatchNorm2d", "type_activation" : "PReLU", "groups" : 64,"output_padding" : [1,0]}
    },
    "skipper":{
      "skip3":{"n_channels" : 128},
      "skip2":{"n_channels" : 96},
      "skip1":{"n_channels" : 64}
    },
    "FSA": {"in_channels": 128, "hidden_size": 2, "out_channels": 128},
  "TGRU": {"in_channels": 128, "hidden_size": 128, "out_channels": 128, "state_size": 33},
}


class CUNet(nn.Module):
    def __init__(self, 
    architecture=architecture_default,
    n_kernel = 384,
    n_dim = 256, 
    n_stride = 128,
    **kwargs
    ):
        super(CUNet, self).__init__()

        self.architecture = architecture


        self.analy = nn.Conv1d(1, n_dim, n_kernel, stride=n_stride, padding=0, dilation=1, bias=False)

        Encoder = DepthwiseNext
        Decoder = TrNext

        ## Encoder
        n_enc = len(self.architecture["encoder"])
        self.n_enc = n_enc
        self.enc = []
        for i in range(n_enc) : 
            module  = Encoder(**self.architecture["encoder"]["enc{}".format(i+1)])
            self.add_module("enc{}".format(i+1),module)
            self.enc.append(module)

        ## Bottleneck
        self.fgru = FSA3Block(**self.architecture["FSA"])
        self.tgru = TGRUBlock(**self.architecture["TGRU"])

        ## Decoder
        self.dec = []
        for i in range(n_enc) : 
            module = Decoder(**self.architecture["decoder"]["dec{}".format(n_enc-i)])
            self.add_module("dec{}".format(n_enc-i),module)
            self.dec.append(module)

        ## Skip Connection
        self.skipper = []
        skipper = SkipGate1

        for i in range(n_enc):
            module = skipper(**self.architecture["skipper"]["skip{}".format(n_enc-i)])
            self.add_module("skip{}".format(n_enc-i),module)
            self.skipper.append(module)

        self.mask = nn.Sigmoid()

        self.synth = nn.ConvTranspose1d(
                in_channels=n_dim,
                out_channels=1,
                kernel_size=n_stride,
                stride=n_stride,
                padding=0, bias=False)

        ## Enroll
        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)
        self.skipper = nn.ModuleList(self.skipper)


    def forward(self, x):
        # x[B,1,L] -> X[B,F,T]
        X = self.analy(x)
        X = X.unsqueeze(1)

        z = X
        skip = []
        for i in range(self.n_enc):
            #print("enc {} : z {}".format(i,z.shape))
            z = self.enc[i](z)
            skip.append(z)

        z = self.fgru(z)
        z, h = self.tgru(z, None)

        for i in range(self.n_enc) : 
            #print("dec {} : z {} : skip{}".format(i,z.shape, skip[self.n_enc-i-1].shape))
            z = self.dec[i](self.skipper[i](z,skip[self.n_enc-i-1]))

        M = self.mask(z)
        #print(M.shape)

        Y = X * M
        Y = Y.squeeze(1)
        #print(Y.shape)
        y = self.synth(Y)

        return y



class CUNet_helper(nn.Module):
    def __init__(self,
        # shape param
        n_kernel = 384,
        n_stride = 128,
        n_dim = 256, 
        ## model param
        architecture=architecture_default,
        **kwargs
    ):
        super(CUNet_helper,self).__init__()

        self.K = n_kernel
        self.S = n_stride

        self.model = CUNet(
            architecture=architecture,
            n_kernel =n_kernel,
            n_dim =n_dim,
            n_stride = n_stride
            )

    def forward(self, x):

        if len(x.shape)  == 2:
            x = x.unsqueeze(1)
        B,C,L = x.shape

        res = self.K - L%self.K

        if res > 0 :
            x = F.pad(x,(0,res,0,0,0,0))

        ## Tobe causal
        if not torch._C._is_tracing() : 
            x = F.pad(x,(self.S*2,0,0,0,0,0))

        y = self.model(x)

        return y[:,0,:L]

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        print("CUNet::Warnning: to_onnx is not implemented yet")
        return

