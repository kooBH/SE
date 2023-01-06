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
    estim_spec = enhanced_mag * phase

    estim = rs.istft(wav,n_fft=hp.audio.n_fft)
    return estim




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
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    os.makedirs(args.dir_output,exist_ok=True)
    list_target = [x for x in glob.glob(os.path.join(args.dir_input,"*.wav"))]

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)
    mono = args.mono

    if mono is None :
        mono = False

    print("sr : {}".format(hp.data.sr))
    print("mono : {}".format(mono))

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

            noisy_wav, _ = rs.load(path,sr=hp.data.sr,mono=mono)
            if len(noisy_wav.shape) == 1 :
                noisy_wav = np.expand_dims(noisy_wav,0)
            data = dataset.get_feature(noisy_wav,hp).to(device)

            n_channel = data.shape[1]

            estim = None
            for i_ch in range(n_channel) : 
                # [B C F T]
                mask = model(data[:,i_ch:i_ch+1,:,:])
                if estim is None : 
                    estim  = model.output(mask,data[:,i_ch:i_ch+1,:,:])[0]
                else :
                    estim  = torch.cat((estim,model.output(mask,data[:,i_ch:i_ch+1,:,:])[0]),0)

            name_target = path.split('/')[-1]
            id_target = name_target.split('.')[0]

            output_wav= mag_domain(hp,noisy_wav,estim)

            sf.write(os.path.join(args.dir_output,"{}".format(name_item)),output_wav.T,hp.data.sr)

