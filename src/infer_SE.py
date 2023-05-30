import torch
import librosa as rs
import soundfile as sf
import argparse
import torchaudio
import os,glob
import numpy as np

from tqdm import tqdm

from utils.hparams import HParam

import common
from common import run,get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default config")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    os.makedirs(args.dir_output,exist_ok=True)
    list_target = [x for x in glob.glob(os.path.join(args.dir_input,"**","*.wav"),recursive=True)]

    device = args.device
    torch.cuda.set_device(device)

    batch_size = 1
    num_workers = 1

    ## load
    model = get_model(hp,device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        #model.load_state_dict(torch.load(args.chkpt, map_location=device)["model"])
        chkpt =  torch.load(args.chkpt, map_location=device)
        if "model" in chkpt : 
            model.load_state_dict(chkpt["model"])
        else : 
            model.load_state_dict(chkpt)

    #### EVAL ####
    model.eval()
    with torch.no_grad():
        for path in tqdm(list_target) : 

            # Path Management
            name_item = path.split('/')[-1]
            name_target = path.split('/')[-1]
            id_target = name_target.split('.')[0]

            path_after_root = path.split(args.dir_input)[1]

            path_before_name = path_after_root.split(name_item)[0]

            ## see : https://docs.python.org/3/library/os.path.html#os.path.join
            if path_before_name == "/" : 
                path_before_name = ""

            noisy_wav, _ = rs.load(path,sr=hp.data.sr)
            noisy_wav = torch.unsqueeze(torch.from_numpy(noisy_wav).float().to(device),0)
            output_wav = model(noisy_wav)
            output_wav = output_wav.detach().cpu().numpy()

            os.makedirs(os.path.join(args.dir_output,path_before_name),exist_ok=True)  
            sf.write(os.path.join(args.dir_output,path_before_name,"{}".format(name_item)),output_wav.T,hp.data.sr)

