import torch
import librosa as rs
import soundfile as sf
import argparse
import torchaudio
import os,glob
import numpy as np

from tqdm import tqdm


from Dataset.DatasetSPEAR import DatasetSPEAR

from utils.hparams import HParam

from UNet.ResUNet import ResUNetOnFreq, ResUNet
from UNet.UNet import UNet

import common
from common import run,get_model


def mag_domain(hp,wav,enhanced_mag):
    enhanced_mag = enhanced_mag.detach().cpu().numpy()
    spec = rs.stft(wav,n_fft=hp.audio.n_fft)
    mag,phase = rs.magphase(spec)

    estim_spec = enhanced_mag * np.exp(phase*1j)

    wav = rs.istft(estim_spec,n_fft=hp.audio.n_fft)
    return wav

def infer(model,data,input, hp):
    n_channel = data.shape[1]

    estim = None

    data = torch.unsqueeze(data,0)

    # [B C F T]
    mask = model(data[:,:,:,:])
    if estim is None : 
        estim  = model.output(mask,data[:,:,:,:])[0]
    else :
        estim  = torch.cat((estim,model.output(mask,data[:,:,:,:])[0]),0)


    if hp.model.mag_only : 
        output_wav= mag_domain(hp,input,estim[0])

    else :
        estim = estim.detach().cpu().numpy()
        estim = estim[0] + estim[1]*1j
        output_wav  = rs.istft(estim,n_fft=hp.audio.n_fft)

    return output_wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default config")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--mono', action=argparse.BooleanOptionalAction)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    parser.add_argument('--sr',type=int,required=False,default=16000)
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    os.makedirs(args.dir_output,exist_ok=True)
    list_target = [x for x in glob.glob(os.path.join(args.dir_input,"**","*.wav"),recursive=True)]

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)
    mono = args.mono

    if mono is None :
        mono = False
    if hp.model.use_cdr : 
        print("ERROR::no CDR")
        exit(-1)

    print("sr : {}".format(hp.data.sr))
    print("mono : {}".format(mono))
    print("mag : {}".format(hp.model.mag_only))

    batch_size = 1
    num_workers = 1

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version

    model = get_model(hp,device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        #model.load_state_dict(torch.load(args.chkpt, map_location=device)["model"])
        chkpt =  torch.load(args.chkpt, map_location=device)
        if "model" in chkpt : 
            model.load_state_dict(chkpt["model"])
        else : 
            model.load_state_dict(chkpt)

    if hp.task == "SPEAR":
        dataset = DatasetSPEAR

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

            noisy_wav, _ = rs.load(path,sr=hp.data.sr,mono=mono)
            #print("input {} | {}".format(path,noisy_wav.shape))

            # norm
            #if hp.model.normalize : 
            #    noisy_wav = noisy_wav/np.max(np.abs(noisy_wav)+1e-7)

            if len(noisy_wav.shape)==1 or mono : 
                data = dataset.get_feature(noisy_wav,hp).to(device)
                output_wav = infer(model,data,noisy_wav,hp)
            else :
                output_wav = []
                for ch in range(noisy_wav.shape[0]) : 
                    data = dataset.get_feature(noisy_wav[ch],hp).to(device)
                    output_wav.append(infer(model,data,noisy_wav[ch],hp))
                output_wav = np.array(output_wav)

            os.makedirs(os.path.join(args.dir_output,path_before_name),exist_ok=True)  

            if hp.data.sr != args.sr : 
                output_wav = rs.resample(output_wav,hp.data.sr,args.sr)
                sf.write(os.path.join(args.dir_output,path_before_name,"{}".format(name_item)),output_wav.T,args.sr)

