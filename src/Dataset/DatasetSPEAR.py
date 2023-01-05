import os
from glob import glob
import torch
import librosa as rs
import common
import numpy as np
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

"""
D1 : real data, talker in data
D2 : sim data, no talkker in data
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

        # ex : /home/data/kbh/SPEAR_seg/Train/noisy
        self.list_noisy += glob(os.path.join(hp.data.root,train_dev,"noisy","*.wav"))

        self.len_data = hp.model.len_sec * hp.data.sr

    @staticmethod
    def get_feature(wav,hp):
        data = rs.stft(wav,n_fft=hp.audio.n_fft)
        #data = np.log(np.abs(data)) + 1e-7
        data = np.abs(data)
        data = torch.from_numpy(data)
        data = torch.unsqueeze(data,0)

        return data

    def __getitem__(self, idx):
        ## Path Data
        path_noisy = self.list_noisy[idx]
        name_noisy = path_noisy.split("/")[-1]

        # ex : /home/data/kbh/SPEAR_seg/Train/clean/Dataset_2_Session_5_01_ch_1_seg_240000.wav
        id_ch = np.random.randint(2)
        name_clean = "_".join(name_noisy.split("_")[:6])+"_"+str(id_ch)+"_"+"_".join(name_noisy.split("_")[7:])

        path_clean = os.path.join(self.hp.data.root,self.train_dev,"clean",name_clean)

        ## Load data
        noisy = rs.load(path_noisy,sr=self.hp.data.sr)[0]
        clean = rs.load(path_clean,sr=self.hp.data.sr)[0]

        # cut
        idx_start = np.random.randint(len(noisy)-self.len_data)

        noisy = noisy[idx_start:idx_start + self.len_data]
        clean = clean[idx_start:idx_start + self.len_data]

        noisy= noisy/(np.max(np.abs(noisy))+1e-7)
        clean= clean/(np.max(np.abs(clean))+1e-7)

        # fea7re
        noisy_mag = DatasetSPEAR.get_feature(noisy,self.hp)
        clean_mag = DatasetSPEAR.get_feature(clean,self.hp)

        data={}
        data["noisy_mag"] = noisy_mag
        data["clean_mag"] = clean_mag

        return data

    def __len__(self):
        return len(self.list_noisy)
