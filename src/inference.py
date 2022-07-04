import torch
import librosa
import soundfile as sf
import argparse
import torchaudio
import os,glob
import numpy as np

from tqdm import tqdm

from utils.hparams import HParam

from UNet.UNet import UNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--mono', action=argparse.BooleanOptionalAction)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    os.makedirs(args.dir_output,exist_ok=True)

    list_target = [x for x in glob.glob(os.path.join(args.dir_input,"*.wav"))]

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)
    mono = args.mono

    batch_size = 1
    num_workers = 1

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version

    model = UNet(
        device=device
    ).to(device)

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
            noisy_wav, _ = librosa.load(path,sr=8000,mono=mono)

            noisy_wav = torch.from_numpy(noisy_wav)
            noisy_spec =  torch.stft(noisy_wav,n_fft=512,return_complex=True,center=True)

            len_orig = noisy_spec.shape[-1]
            need =  int(16*np.floor(len_orig/16)+16) - len_orig
            noisy_spec = torch.nn.functional.pad(noisy_spec,(0,need))

            noisy_mag = torch.abs(noisy_spec)
            noisy_phase = torch.angle(noisy_spec)
        
            noisy_mag = torch.unsqueeze(noisy_mag,dim=0).to(device)
            #noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
            noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

            output_wav = []

            for i_ch in range(n_channel) : 
                mask = model(noisy_mag[:,i_ch:i_ch+1,:,:])[0]

                # masking
                estim_mag = noisy_mag*mask
                estim_spec = estim_mag * (noisy_phase*1j).to(device)

                # inverse
                estim_wav = torch.istft(estim_spec[:,0,:,:],n_fft = 512)

                output_wav.append(estim_wav[0].cpu().detach().numpy())

            name_target = path.split('/')[-1]
            id_target = name_target.split('.')[0]

            output_wav = np.array(output_wav)

            sf.write(os.path.join(args.dir_output,"{}.wav".format(id_target)),output_wav.T,8000)

