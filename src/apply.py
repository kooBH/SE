import torch
import torch.nn as nn
import torch.nn.functional as F

#define custom_atan2 to support onnx conversion
def custom_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / (x + 1e-6))
    ans += ((y > 0) & (x < 0)) * pi
    ans -= ((y < 0) & (x < 0)) * pi
    ans *= 1 - ((y > 0) & (x == 0)) * 1.0
    ans += ((y > 0) & (x == 0)) * (pi / 2)
    ans *= 1 - ((y < 0) & (x == 0)) * 1.0
    ans += ((y < 0) & (x == 0)) * (-pi / 2)
    return ans

"""
from https://github.com/echocatzh/MTFAA-Net/blob/main/mtfaa.py
part of MTFAANet
"""
class MEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=4, mag_f_dim=3,eps=1e-7):
        super(MEA, self).__init__()
        self.mag_mask = nn.Conv2d(
            in_channels, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = torch.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim

        self.eps = eps
    
    def forward(self, x, z):
        # x : input STFT [B,F,T,2]
        # z : output feature [B, C, F, T]
        mag = torch.norm(x, dim=-1)
        pha = custom_atan2(x[..., 1], x[..., 0])

        # stage 1
        mag_mask = self.mag_mask(z)
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.relu()
        mag = mag.sum(dim=1)

        # stage 2
        real_mask = self.real_mask(z).squeeze(1)
        imag_mask = self.imag_mask(z).squeeze(1)

        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, self.eps))
        pha_mask = custom_atan2(imag_mask+self.eps, real_mask+self.eps)
        real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)