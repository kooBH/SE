import torch
import librosa
import soundfile as sf
import argparse
import torchaudio
import os,glob
import numpy as np

from tqdm import tqdm

from utils.hparams import HParam

from UNet.ResUNet import ResUNetOnFreq, ResUNet
from UNet.UNet import UNet


from common import run,get_model

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
    parser.add_argument('--DB', action=argparse.BooleanOptionalAction)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    os.makedirs(args.dir_output,exist_ok=True)
    if args.DB : 
        list_target = [x for x in glob.glob(os.path.join(args.dir_input,"*","*","noisy","*.wav"))]
    else :
        list_target = [x for x in glob.glob(os.path.join(args.dir_input,"*.wav"))]

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)
    mono = args.mono

    batch_size = 1
    num_workers = 1

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version



    model = get_model(hp).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    if mono : 
        n_channel = 1
    else :
        n_channel = 7

    #### EVAL ####
    model.eval()
    with torch.no_grad():
        for path in tqdm(list_target) : 

            # Path Management
            if args.DB : 
                path_from_in = path.split(args.dir_input)[1]
                name_item = path.split('/')[-1]
                dir_from_in = path_from_in.split(name_item)[0]
                mid_dir_out = dir_from_in.replace("noisy","estim")

                os.makedirs(os.path.join(args.dir_output,mid_dir_out),exist_ok=True)
            else :
                name_item = path.split('/')[-1]


            noisy_wav, _ = librosa.load(path,sr=hp.data.sr,mono=mono)

            noisy_wav = torch.from_numpy(noisy_wav)
            noisy_spec =  torch.stft(noisy_wav,n_fft=hp.data.n_fft,return_complex=True,center=True)

            len_orig = noisy_spec.shape[-1]
            need =  int(16*np.floor(len_orig/16)+16) - len_orig
            noisy_spec = torch.nn.functional.pad(noisy_spec,(0,need))

            if hp.model.mag_only : 
                noisy_mag = torch.abs(noisy_spec)
                noisy_phase = torch.angle(noisy_spec)
            
                noisy_mag = torch.unsqueeze(noisy_mag,dim=0).to(device)
                if mono : 
                    noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
                noisy_phase = torch.unsqueeze(noisy_phase,dim=0)
                in_feature = noisy_mag
            else : 
                in_feature= torch.permute(torch.view_as_real(noisy_spec),(2,0,1))
                in_feature = torch.unsqueeze(in_feature,dim=0).to(device)

            output_wav = []
            for i_ch in range(n_channel) : 
                # TODO fix for multichannel
                #mask = model(in_feature[:,i_ch:i_ch+1,:,:])[0]
                mask = model(in_feature[:,:,:,:])[0]


                # masking
                if hp.model.mag_only : 
                    estim_mag = noisy_mag*mask
                    estim_spec = estim_mag * (noisy_phase*1j).to(device)
                else :
                    estim_spec = mask * in_feature
                    estim_spec = estim_spec[:,0:1,:,:]+estim_spec[:,1:2,:,:]*1j

                # inverse
                estim_wav = torch.istft(estim_spec[:,0,:,:],n_fft = hp.data.n_fft)

                output_wav.append(estim_wav[0].cpu().detach().numpy())

            name_target = path.split('/')[-1]
            id_target = name_target.split('.')[0]

            output_wav = np.array(output_wav)

            # norm
            output_wav = output_wav/np.max(np.abs(output_wav))

            if args.DB : 
                sf.write(os.path.join(args.dir_output,mid_dir_out,"{}".format(name_item)),output_wav.T,hp.data.sr)
            else :
                sf.write(os.path.join(args.dir_output,"{}".format(name_item)),output_wav.T,hp.data.sr)

