from common import run,get_model
from utils.hparams import HParam
import torch

device = "cpu"

path_default = "../config/DDUNet/default.yaml"
path_config = "../config/DDUNet/v0.yaml"

#hp = HParam(path_config,path_default,merge_except=["architecture"])
hp = HParam(path_config,path_default)
model = get_model(hp,device=device)

x = torch.rand(2,49152)

y,h = model(x)
print("{} -> {}".format(x.shape,y.shape))
