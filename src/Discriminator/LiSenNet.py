# https://github.com/hyyan2k/LiSenNet/blob/main/model/discriminator/discriminator.py

import torch
import torch.nn as nn


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Discriminator(nn.Module):
    def __init__(self, ndf=16, in_channel=2, **kwrags):
        super().__init__()
        norm_f = nn.utils.spectral_norm
        self.layers = nn.ModuleList([
            nn.Sequential(
                norm_f(nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf, affine=True),
                nn.PReLU(ndf),
            ),
            nn.Sequential(
                norm_f(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf * 2, affine=True),
                nn.PReLU(2 * ndf),
            ),
            nn.Sequential(
                norm_f(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf * 4, affine=True),
                nn.PReLU(4 * ndf),
            ),
            nn.Sequential(
                norm_f(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf * 8, affine=True),
                nn.PReLU(8 * ndf),
            ),
            nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                norm_f(nn.Linear(ndf * 8, ndf * 4)),
                nn.Dropout(0.3),
                nn.PReLU(4 * ndf),
                norm_f(nn.Linear(ndf * 4, 1)),
                LearnableSigmoid(1),
            ),
        ])

    def forward(self, x, y):
        assert x.ndim == 3
        xy = torch.stack([x, y], dim=1)
        outs = []
        for layer in self.layers:
            xy = layer(xy)
            outs.append(xy)
        return outs

class DiscriminatorWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.discriminator = Discriminator(kwargs)

    def forward(self, x, y):

        X = torch.stft(x, n_fft=512, hop_length=128, win_length=512, return_complex=True)
        Y = torch.stft(y, n_fft=512, hop_length=128, win_length=512, return_complex=True)

        X_mag = torch.sqrt(X.real**2 + X.imag**2)
        Y_mag = torch.sqrt(Y.real**2 + Y.imag**2)

        return self.discriminator(X_mag, Y_mag)
