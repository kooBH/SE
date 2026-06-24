import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()

        self.layer = nn.Conv2d(2,2, kernel_size=3, padding=1)

    def forward(self, x):
        #  x : [B,F,T,2]

        X = torch.stack((x[...,0], x[...,1]), dim=1)  #  X : [B,2,F,T]

        X = self.layer(X)  # Y : [B,2,F,T]
        Y = torch.stack((X[:,0], X[:,1]), dim=-1)  # Y : [B,F,T,2]

        return Y

class ModelWrapper(nn.Module):
    def __init__(self,
            n_fft: int = 512,
            hop_size : int = 256,
            input_compression: float = 0.3,
            **kwargs
            ):
        super().__init__()
        self.model = Model(**kwargs)

        eps = 1e-7

        # Compressed STFT and discard_last_freq_bin
        self.frame_size = n_fft
        self.hop_size = hop_size
        self.compression = input_compression
        self.eps = eps
        
        self.window = torch.zeros(self.frame_size)
        if self.frame_size // self.hop_size == 4 : 
            n = torch.arange(self.frame_size)
            self.window = 0.5 * (1.0 - torch.cos(2.0 * np.pi * n / self.frame_size))
            
            energy_sum = torch.sum(self.window ** 2) / self.hop_size
            self.window /= torch.sqrt(energy_sum)
        elif self.frame_size // self.hop_size == 2 :
            n = torch.arange(self.frame_size)
            self.window = torch.sin(np.pi * (n + 0.5) / self.frame_size)
            
            energy_sum = torch.sum(self.window ** 2) / self.hop_size
            self.window /= torch.sqrt(energy_sum)
        else :
            raise RuntimeError(f"Not supported frame_size // hop_size {self.frame_size}//{self.hop_size}")

    def _to_spec(self,x):
        B,L = x.shape
        # X : [B,F,T,2]
        X = torch.stft(x, n_fft = self.frame_size, hop_length = self.hop_size, window=self.window.to(x.device),center=True, return_complex=False)

        # Magnitudfe Compression
        mag = torch.linalg.norm(X, dim=-1, keepdim=True).clamp(min=self.eps)
        X = X * mag.pow(self.compression - 1.0)
        return X

    def _to_signal(self, Y, length=None):
        Y = Y[...,0] + 1j * Y[...,1]  

        # Magnitude Decompression
        mag_compressed = Y.abs()
        Y = Y* mag_compressed.pow((1.0 / self.compression) - 1.0)

        y = torch.istft(Y, self.frame_size, self.hop_size, self.frame_size, self.window.to(Y.device), center=True, length=length)
        
        return y

    def forward(self, x):
        B,L = x.shape
        
        X = self._to_spec(x)
        Y = self.model(X)

        y = self._to_signal(Y, length=L)
        return y
