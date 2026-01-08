import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import math

### Estimating

# LSNR Estimator : C,F -> 1
class lsnr_estimator1(nn.Module):
    def __init__(self,hidden_size=256, **kwargs) : 
        super(lsnr_estimator1,self).__init__()
        self.linear = nn.Linear(hidden_size,1)
        self.activation = nn.Tanh()

    def forward(self,x):
        # x : B,C,T,F -> B*T,C*F
        B, C, T, F = x.shape
        x = x.permute(0,2,1,3)
        x = x.reshape(B*T, C*F)

        lsnr = self.linear(x)
        lsnr = self.activation(lsnr)

        # lsnr : B*T,1 -> B,T,1
        lsnr = lsnr.reshape(B, T, 1)
        return lsnr
    
class GroupedLinear(nn.Module):
    input_size: int
    hidden_size: int
    groups: int
    def __init__(self, 
        input_size: int, 
        hidden_size: int,
        groups: int = 1
        ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        # einsum operation ->  iterate bmm over groups
        self.bmm = nn.Linear(self.input_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i in range(self.groups):
            outputs.append(self.bmm(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)

        return output
    
# LSNr Estimator2 : C,F -> C -> 1
class lsnr_estimator2(nn.Module): 
    def __init__(self, hidden_size=29, input_channels=256, **kwargs):
        super(lsnr_estimator2,self).__init__()

        self.enc1 = nn.Conv2d(hidden_size, 1, kernel_size=(1,1), bias=False)
        self.act1 = nn.ReLU()
        self.linear = nn.Linear(input_channels, 1, bias=False)
        self.act2 = nn.Tanh()

    def forward(self,x):
        # x : B,C,T,F -> B,F,T,C
        B, C, T, F = x.shape
        x = x.permute(0,3,2,1)

        x = self.enc1(x)  # B,1,T,C
        x = self.act1(x)
        x = self.linear(x)  # B,1,T,1
        x = self.act2(x)
        x = x.squeeze(1)  # B,T,1

        return x

# LSNR Estimator4 : C,F -> F -> 1
class lsnr_estimator4(nn.Module):
    def __init__(self, input_channels = 256,hidden_size = 29, **kwargs) : 
        super(lsnr_estimator4,self).__init__()
        self.n1 = nn.BatchNorm2d(input_channels)
        self.m1 = nn.Conv2d(input_channels,1,kernel_size=(1,1),bias=False)
        self.a1 = nn.ReLU()
        self.m2 = nn.Linear(hidden_size,1,bias=False)
        self.a2 = nn.Tanh()

    def forward(self,x):
        # x : B,C,T,F -> B,1,T,F
        B, C, T, F = x.shape
        x = self.n1(x)
        x = self.m1(x)
        x = self.a1(x)

        x = x.permute(0,2,1,3)
        x = x.reshape(B*T, -1)  # B*T, C*F
        x = self.m2(x)
        x = self.a2(x)

        # lsnr : B*T,1 -> B,T,1
        lsnr = x.reshape(B, T, 1)
        return lsnr
    
# TODO : LSNR Estimator3 : C,F -> F -> 1

# TODO : LSNR Estimator4 : C,F -> F -> freq-wise lsnr

### Maksing
    
# V1, mask is calculated with lsnr
class SNRAwareMagMaskingV1(nn.Module):
    def __init__(self,hidden_size, lsnr_location="None", **kwargs) : 
        super(SNRAwareMagMaskingV1,self).__init__()
        self.activation = nn.Sigmoid()
        if lsnr_location == "None":
            self.fusion = nn.Linear(hidden_size,hidden_size)
        else :
            self.fusion = nn.Linear(hidden_size+1,hidden_size)

    def forward(self,X, z, lsnr=None) : 
        # X : B,2,T,F
        # z : B,1,T,F                    
        # lsnr : B,T,1

        if lsnr is None :
            m = self.fusion(z)
            m = self.activation(m)
        else :
            lsnr = lsnr.unsqueeze(1)
            m = self.fusion(torch.cat((z, lsnr), dim=-1))
            m = self.activation(m)
        m = m.squeeze(1)  # B,T,F

        mag_X = torch.sqrt(X[:,0]**2 + X[:,1]**2 + 1e-8)
        phase_X = torch.atan2(X[:,1], X[:,0])
        #print(f"SNRAwareMagMasking::mag : {mag_X.shape}, phase : {phase_X.shape}, m {m.shape}")
        mag_Y = mag_X * m
        re_Y = mag_Y * torch.cos(phase_X)
        im_Y = mag_Y * torch.sin(phase_X)
        Y = torch.stack((re_Y, im_Y), dim=1)

        #print(f"SNRAwareMagMasking::X : {X.shape}, Y : {Y.shape}")

        return Y
    
# V2, mask weight is calculated from lsnr
class SNRAwareMagMaskingV2(nn.Module):
    def __init__(self,hidden_size, lsnr_location="None", **kwargs) : 
        super(SNRAwareMagMaskingV2,self).__init__()
        self.activation = nn.Sigmoid()
        self.fusion = nn.Linear(hidden_size,hidden_size)
        self.weight = nn.Linear(1,hidden_size)

    def forward(self,X, z, lsnr=None) : 
        # X : B,2,T,F
        # z : B,1,T,F                    
        # lsnr : B,T,1

        m = self.fusion(z)
        m = self.activation(m)
        if lsnr is not None :
            a = self.weight(lsnr)  # B,T,F
            a = self.activation(a)
        else :
            a = 1
        m = m.squeeze(1)  # B,T,F

        mag_X = torch.sqrt(X[:,0]**2 + X[:,1]**2 + 1e-8)
        phase_X = torch.atan2(X[:,1], X[:,0])
        #print(f"SNRAwareMagMasking::mag : {mag_X.shape}, phase : {phase_X.shape}, m {m.shape}")
        mag_Y =  (1 + m - a) * mag_X
        re_Y = mag_Y * torch.cos(phase_X)
        im_Y = mag_Y * torch.sin(phase_X)
        Y = torch.stack((re_Y, im_Y), dim=1)

        #print(f"SNRAwareMagMasking::X : {X.shape}, Y : {Y.shape}")

        return Y
    
# V2_1, mask weight is calculated from lsnr
class SNRAwareMagMaskingV2_1(nn.Module):
    def __init__(self,hidden_size, lsnr_location="None", **kwargs) : 
        super(SNRAwareMagMaskingV2_1,self).__init__()
        self.activation = nn.Sigmoid()
        self.weight = nn.Linear(1,hidden_size)

    def forward(self,X, z, lsnr=None) : 
        # X : B,2,T,F
        # z : B,1,T,F                    
        # lsnr : B,T,1

        m = self.activation(z)
        if lsnr is not None :
            a = self.weight(lsnr)  # B,T,F
            a = self.activation(a)
        m = m.squeeze(1)  # B,T,F

        mag_X = torch.sqrt(X[:,0]**2 + X[:,1]**2 + 1e-8)
        phase_X = torch.atan2(X[:,1], X[:,0])
        #print(f"SNRAwareMagMasking::mag : {mag_X.shape}, phase : {phase_X.shape}, m {m.shape}")
        mag_Y =  (1 + m - a) * mag_X
        re_Y = mag_Y * torch.cos(phase_X)
        im_Y = mag_Y * torch.sin(phase_X)
        Y = torch.stack((re_Y, im_Y), dim=1)

        #print(f"SNRAwareMagMasking::X : {X.shape}, Y : {Y.shape}")

        return Y