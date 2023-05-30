import os
from os.path import join
from glob import glob
import torch
import librosa as rs
import numpy as np
import random
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

class DatasetDNS(torch.utils.data.Dataset):
    def __init__(self,hp,is_train=True):
        self.hp = hp
        self.is_train = is_train

        if is_train : 
            self.list_clean = glob(join(hp.data.clean,"**","*.wav"),recursive=True)
            self.list_noise = glob(join(hp.data.noise,"**","*.wav"),recursive=True)
            self.list_RIR   = glob(join(hp.data.RIR,"**","*.wav"),recursive=True)
        else :
            self.list_noisy = glob(join(hp.data.dev.root,"*","noisy","*.wav"),recursive=True)
            self.list_RIR   = None
            self.list_clean = None

            self.eval={}
            self.eval["with_reverb"]={}
            self.eval["no_reverb"]={}

            self.eval["with_reverb"] = glob(join(hp.data.dev.root,"with_reverb","noisy","*.wav"),recursive=True)

            self.eval["no_reverb"] = glob(join(hp.data.dev.root,"no_reverb","noisy","*.wav"),recursive=True)



        self.range_SNR = hp.data.SNR
        self.target_dB_FS = -25
        self.target_dB_FS_floating_value = 10

        self.len_data = hp.data.len_data
        self.n_item = hp.data.n_item

        self.sr = hp.data.sr

        if is_train : 
            print("DatasetDNS[train:{}] | len : {} | clean {} | noise : {} | RIR : {}".format(is_train,self.n_item,len(self.list_clean),len(self.list_noise),len(self.list_RIR)))
        else :
            print("DatasetDNS[train:{}] | len : {} | noisy : {}".format(is_train,self.n_item,len(self.list_noisy)))

    def match_length(self,wav,idx_start=None) : 
        if len(wav) > self.len_data : 
            left = len(wav) - self.len_data
            if idx_start is None :
                idx_start = np.random.randint(left)
            wav = wav[idx_start:idx_start+self.len_data]
        elif len(wav) < self.len_data : 
            shortage = self.len_data - len(wav) 
            wav = np.pad(wav,(0,shortage))
        return wav, idx_start

    @staticmethod
    def norm_amplitude(y, scalar=None, eps=1e-6):
        if not scalar:
            scalar = np.max(np.abs(y)) + eps

        return y / scalar, scalar

    @staticmethod
    def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    @staticmethod
    def is_clipped(y, clipping_threshold=0.999):
        return any(np.abs(y) > clipping_threshold)

    def mix(self,clean,noise,rir,eps=1e-7):
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]
            clean = signal.fftconvolve(clean, rir)[:len(clean)]

        clean, _ = self.norm_amplitude(clean)
        clean, _, _ = self.tailor_dB_FS(clean, self.target_dB_FS)
        clean_rms = (clean ** 2).mean() ** 0.5

        noise, _ = self.norm_amplitude(noise)
        noise, _, _ = self.tailor_dB_FS(noise, self.target_dB_FS)
        noise_rms = (noise ** 2).mean() ** 0.5

        SNR = noisy_target_dB_FS = np.random.randint(
            self.range_SNR[0],self.range_SNR[1]
        )

        snr_scalar = clean_rms / (10 ** (SNR / 20)) / (noise_rms + eps)
        noise *= snr_scalar
        noisy = clean + noise

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            self.target_dB_FS - self.target_dB_FS_floating_value,
            self.target_dB_FS + self.target_dB_FS_floating_value
        )

        # rescale noisy RMS
        noisy, _, noisy_scalar = self.tailor_dB_FS(noisy, noisy_target_dB_FS)
        clean *= noisy_scalar

        if self.is_clipped(noisy):
            noisy_scalar = np.max(np.abs(noisy)) / (0.99 - eps)  # same as divide by 1
            noisy = noisy / noisy_scalar
            clean = clean / noisy_scalar

        return noisy, clean
    
    def get_clean_dev(self,path_noisy):
        path_after_root = path_noisy.split(self.hp.data.dev.root)[-1]
        dev_type = path_after_root.split("/")[0]

        fid = path_noisy.split("_")[-1]
        fid = fid.split(".")[0]

        path_clean = os.path.join(self.hp.data.dev.root,dev_type,"clean","clean_fileid_{}.wav".format(fid))

        return path_clean

    def __getitem__(self, idx):
        
        if self.is_train : 
            # sample clean
            path_clean = random.choice(self.list_clean)
            clean = rs.load(path_clean,sr=self.sr)[0]

            # sample noise
            path_noise = random.choice(self.list_noise)
            noise = rs.load(path_noise,sr=self.sr)[0]

            if self.hp.data.use_RIR : 
                # sample RIR
                path_RIR = random.choice(self.list_RIR)
                RIR = rs.load(path_RIR,sr=self.sr)[0]
            else :
                RIR = None

            ## Length Match
            clean,_ = self.match_length(clean)
            noise,_ = self.match_length(noise)

            # mix 
            noisy,clean = self.mix(clean,noise,RIR)
        else :
            path_noisy = self.list_noisy[idx]
            path_clean = self.get_clean_dev(path_noisy)
            

            clean = rs.load(path_clean,sr=self.sr)[0]
            noisy = rs.load(path_noisy,sr=self.sr)[0]

            clean,idx_start = self.match_length(clean)
            noisy,_ = self.match_length(noisy,idx_start)

        data = {"noisy":noisy,"clean":clean}
        return  data

    def __len__(self):
        if self.is_train : 
            return self.n_item
        else :
            return len(self.list_noisy)
        
    def get_eval(self,idx) : 

        path_reverb = self.eval["with_reverb"][idx]
        path_no_reverb = self.eval["no_reverb"][idx]

        path_clean_reverb = self.get_clean_dev(path_reverb)
        path_clean_no_reverb = self.get_clean_dev(path_no_reverb)

        return [path_reverb,path_clean_reverb],[path_no_reverb,path_clean_no_reverb]

## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("./")
    from utils.hparams import HParam
    hp = HParam("../config/SPEAR/v20.yaml","../config/SPEAR/default.yaml")
  




