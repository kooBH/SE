import os
from glob import glob
import torch
import librosa as rs
import numpy as np
import random
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

"""
D1 : real data, talker in data
D2 : sim data, no talker in data
D3 : sim data, no talker in data 
D4 : sim data, no talker in data

"""

class DatasetSPEAR(torch.utils.data.Dataset):
    def __init__(self,hp,is_train=True):
        self.hp = hp

        self.list_noisy = []
        if is_train : 
            train_dev  = "Train"
        else :
            train_dev  = "Dev"
        self.train_dev = train_dev

        self.use_cdr = hp.model.use_cdr

        if hp.data.augment.clean : 
            self.list_aug_clean = glob(os.path.join(hp.data.augment.root_clean,"**","*.wav"),recursive=True)
        else  :
            self.list_aug_clean = None
        
        if hp.data.augment.noise : 
            self.list_aug_noise = glob(os.path.join(hp.data.augment.root_noise,"**","*.wav"),recursive=True)
        else :
            self.list_aug_noise = None

        if hp.data.augment.RIR : 
            self.list_aug_rir = glob(os.path.join(hp.data.augment.root_RIR,"**","*.wav"),recursive=True)
        else :
            self.list_aug_rir = None

        # ex : /home/data/kbh/SPEAR_seg/Train/noisy/Dataset_1_Session_10_00_4_seg_0.wav
        for i in hp.data.dataset : 
            self.list_noisy += glob(os.path.join(hp.data.root,train_dev,"noisy","Dataset_{}_*.wav".format(i)))

        self.len_data = hp.data.len_sec * hp.data.sr

    @staticmethod
    def get_feature(wav,hp):
        data = rs.stft(wav,n_fft=hp.audio.n_fft)

        #data = np.log(np.abs(data)) + 1e-7
        if hp.model.mag_only : 
            # [F.T]
            data = np.abs(data)
            if hp.model.dB : 
                data = 10*np.log10(data+1e-7)
            data = torch.from_numpy(data)
        else : 
            # [2, F ,T]
            data = np.stack((data.real,data.imag))
            data = torch.from_numpy(data)

        if len(data.shape) < 3 :
            data = torch.unsqueeze(data,0)

        return data

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
    def tailor_dB_FS(y, target_dB_FS, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y

    def __getitem__(self, idx):
        ## Path Data
        path_noisy = self.list_noisy[idx]
        # it should be with out id, there was a mistake
        name_noisy = path_noisy.split("/")[-1]

        ## SPEAR_seg data
        id_ch = np.random.randint(2)
        # Dataset_2_Session_10_00_ch_0_seg_120000.wav
        name_clean = "_".join(name_noisy.split("_")[:6])+"_"+str(id_ch)+"_"+"_".join(name_noisy.split("_")[7:])

        path_clean = os.path.join(self.hp.data.root,self.train_dev,"clean",name_clean)

        ## Load data
        noisy = rs.load(path_noisy,sr=self.hp.data.sr)[0]
        clean = rs.load(path_clean,sr=self.hp.data.sr)[0]

        #noisy = noisy[idx_start:idx_start + self.len_data]
        #clean = clean[idx_start:idx_start + self.len_data]

        noisy,idx_start = self.match_length(noisy)
        clean,idx_start = self.match_length(clean,idx_start)

        aug_clean = None
        if self.hp.data.augment.clean :
            path_aug = random.sample(self.list_aug_clean,1)[0]
            aug_clean = rs.load(path_aug,sr=self.hp.data.sr)[0]
            aug_clean,idx_start = self.match_length(aug_clean)
            noisy += aug_clean
            clean += aug_clean

        if self.hp.data.augment.RIR :
            raise Exception("ERROR::DatasetSPEAR.py::RIR augmentation is not ready...")
            path_rir = random.sample(self.list_aug_rir,1)[0]
            aug_rir = rs.load(path_aug,sr=self.hp.data.sr,mono = False)[0]
            if aug_rir.ndim > 1 : 
                ch_rir = np.random.randint(aug_rir.shape[0])
                aug_rir = aug_rir[ch_rir,:]

            clean = signal.fftconvolve(clean, aug_rir)[:len(clean)]

        if self.hp.data.augment.noise :
            path_aug = random.sample(self.list_aug_noise,1)[0]
            aug_noise = rs.load(path_aug,sr=self.hp.data.sr)[0]
            aug_noise,idx_start = self.match_length(aug_noise)

            # SNR
            SNR = np.random.uniform(self.hp.data.augment.SNR[0],self.hp.data.augment.SNR[1],1)[0]
            clean_rms = (clean** 2).mean() ** 0.5
            noise_rms = (aug_noise ** 2).mean() ** 0.5
            snr_scalar = clean_rms / (10 ** (SNR / 20)) / (noise_rms + 1e-7)
            aug_noise *= snr_scalar

            noisy += aug_noise

        if self.hp.model.normalize :  
            noisy= noisy/(np.max(np.abs(noisy))+1e-7)
            clean= (clean+1e-7)/(np.max(np.abs(clean))+1e-7)

        elif self.hp.data.augment.noise : 
            dBFS_clean = np.random.uniform(self.hp.data.augment.dBFS,1)[0]
            dBFS_noisy = np.random.uniform(self.hp.data.augment.dBFS,1)[0]
            noisy = DatasetSPEAR.tailor_dB_FS(noisy,dBFS_noisy)
            clean = DatasetSPEAR.tailor_dB_FS(clean,dBFS_clean)

        # feature
        feat_noisy= DatasetSPEAR.get_feature(noisy,self.hp)
        feat_clean= DatasetSPEAR.get_feature(clean,self.hp)

        data={}
        data["noisy"] = feat_noisy
        data["clean"] = feat_clean

        data["noisy_wav"] = noisy
        data["clean_wav"] = clean

        return data

    def __len__(self):
        return len(self.list_noisy)



## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("./")
    from utils.hparams import HParam
    hp = HParam("../config/SPEAR/v20.yaml","../config/SPEAR/default.yaml")
    db = DatasetSPEAR(hp,is_train=False)

    db[0]




