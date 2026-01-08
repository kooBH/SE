import librosa as rs
import soundfile as sf
from tqdm.auto import tqdm
import torch
import  os, glob

import argparse

from utils.hparams import HParam
from common import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="data/processed1_mpB-PF_upScale_4.wav")
parser.add_argument("--output", type=str, default="data/output")
parser.add_argument("--chkpt", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--default", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

root_chkpt = args.chkpt
name_input = os.path.basename(args.input).replace(".wav","")
base_name = os.path.basename(args.chkpt).replace(".pt","")
hp = HParam(args.config,args.default,merge_except=["architecture"])
device = args.device

print(f"== Processing ==")
print(f"Model : {args.tag}")
print(f"Checkpoint : {base_name}")
print(f"Sample : {name_input}")

os.makedirs(f"{args.output}", exist_ok=True)

m = get_model(hp,device)
m.load_state_dict(torch.load(args.chkpt))
m.eval()

x = rs.load(f"data/{name_input}.wav",sr=16000)[0]

x = torch.tensor(x).unsqueeze(0)
x = x.to(device)

with torch.no_grad():
    y = m(x)
    y = y.cpu()
    y = y.squeeze(0).detach().numpy()

sf.write(f"{args.output}/{name_input}_{args.tag}_{base_name}.wav",y,16000)