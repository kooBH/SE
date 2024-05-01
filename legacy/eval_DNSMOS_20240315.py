# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import argparse
import concurrent.futures
import glob
import os

import torch
import librosa as rs
import soundfile as sf
from utils.hparams import HParam
from common import run,get_model

import numpy as np
import pandas as pd
import soundfile as sf
from requests import session
from DNSMOS.DNSMOS import ComputeScore
from tqdm import tqdm


def eval_DNSMOS(dir_in,csv_out,sr):
    models = glob.glob(os.path.join(dir_in, "*"))
    audio_clips_list = []

    primary_model_path = os.path.join("src",'DNSMOS', 'sig_bak_ovr.onnx')
    p808_model_path = os.path.join("src",'DNSMOS', 'model_v8.onnx')

    compute_score = ComputeScore(primary_model_path,p808_model_path,sr)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(dir_in, "*.wav"))
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(dir_in, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)
    csv_path = csv_out
    df.to_csv(csv_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('-i', "--dir_in",type=str,required=True)
    parser.add_argument('-o', "--dir_out",type=str,required=True)
    parser.add_argument('-s', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument("--sr", type=int ,default=16000)
    args = parser.parse_args()


    # PARSING
    list_in = glob.glob(os.path.join(args.dir_in, "*.wav"))

    os.makedirs(args.dir_out,exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path),exist_ok=True)


    # PARAM
    hp = HParam(args.config,args.default)
    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    # MODEL
    model = get_model(hp,device=device)
    print('NOTE::Loading pre-trained model : '+ args.chkpt)
    try : 
        model.load_state_dict(torch.load(args.chkpt, map_location=device)["model"])
    except KeyError :
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    # ESTIM
    model.eval()
    with torch.no_grad():
        for path_in in tqdm(list_in) : 

            x,_ = rs.load(path_in,sr=hp.data.sr)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            y = model(x)
            y = y.squeeze(0).cpu().numpy()

            path_out = os.path.join(args.dir_out, os.path.basename(path_in))
            sf.write(path_out,y,hp.data.sr)

    # EVAL
    eval_DNSMOS(args.dir_out,args.csv_path,args.sr)


