import torch
import os

from Dataset.DatasetDNS import DatasetDNS
from utils.hparams import HParam

from common import run,get_model

path_default = "../config/mpANC_v0/default.yaml"
path_config  = "../config/mpANC_v0/v22.yaml"
path_chkpt   = "../chkpt/mpANC_v0_v22.pt"

device = "cpu"

hp = HParam(path_config,path_default,merge_except=["architecture"])
model = get_model(hp,device=device)
model.eval()

try : 
    model.load_state_dict(torch.load(path_chkpt, map_location=device)["model"])
except KeyError :
    model.load_state_dict(torch.load(path_chkpt, map_location=device))

backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

print_model_size(model) 
print_model_size(model_static_quantized) 


test_dataset  = DatasetDNS(hp,is_train=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,num_workers=1,pin_memory=True)

def run(
    hp,
    data,
    model,
    ): 
    feature = data["noisy"].to(torch.qint8)
    estim = model(feature)
    return estim

with torch.no_grad():
    for j, (data) in enumerate(test_loader):
        estim = run(hp,data,model)
        estim_quant = run(hp,data,model_static_quantized)
        break

