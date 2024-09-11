from utils.Loss import *
import torch

x = torch.rand(2,16000)
y = torch.rand(2,16000)

loss = MultiscaleAntiWrappingLoss([256,512,1024])

print(loss(x,y))


x = x[..., : -(x.shape[-1] % 1024)]
y = y[..., : -(y.shape[-1] % 1024)]

loss3 = MultiscaleSpectrogramLoss([256,512,1024])
print(loss3(x,y))

loss2 = MultiscaleCosSDRLoss([256,512,1024])
print(loss2(x,y))



