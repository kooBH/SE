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
        self.list_noisy += glob(os.path.join(hp.data.root_noisy,train_dev,"Dataset_2","Microphone_Array_Audio","*","*.wav"))
        self.list_noisy += glob(os.path.join(hp.data.root_noisy,train_dev,"Dataset_3","Microphone_Array_Audio","*","*.wav"))
        self.list_noisy += glob(os.path.join(hp.data.root_noisy,train_dev,"Dataset_4","Microphone_Array_Audio","*","*.wav"))

        self.len_data = hp.model.len_sec * hp.data.sr

    def __getitem__(self, idx):
        ## Path Data

        # ex : /home/data/kbh/SPEAR_INPUT/Train/Dataset_2/Microphone_Array_Audio/Session_1/array_D1_S1_M00.wav
        path_noisy = self.list_noisy[idx]

        # Dataset_2
        dir_dataset = path_noisy.split("/")[-4]
        # Session_1
        dir_session = path_noisy.split("/")[-2]
        # 00
        id_utt = path_noisy.split("_")[-1][1:3]

        # ex : /home/data/kbh/SPEAR_TARGET/Train/Dataset_2/Reference_Audio/Session_1/00/ref_D2_S1_M00_ID4.wav
        # ex : /home/data/kbh/SPEAR_TARGET/Train/Dataset_2/Reference_Audio/Session_1/00/ref_D2_S1_M00_ID6.wav
        
        list_clean = glob(os.path.join(self.hp.data.root_target,self.train_dev,dir_dataset,"Reference_Audio",dir_session,id_utt,"*.wav"))

        ## Load data
        idx_noisy = np.random.randint(6)
        idx_clean = np.random.randint(2)

        noisy = rs.load(path_noisy,sr=self.hp.data.sr,mono=False,res_type="fft")[0][idx_noisy,:]
        clean = None

        for path_clean in list_clean : 
            if clean is None  : 
                clean = rs.load(path_clean,sr=self.hp.data.sr,mono=False,res_type="fft")[0][idx_clean,:]
            else :
                clean += rs.load(path_clean,sr=self.hp.data.sr,mono=False,res_type="fft")[0][idx_clean,:]

        # cut
        idx_start = np.random.randint(len(noisy)-self.len_data)

        noisy = noisy[idx_start:idx_start + self.len_data]
        clean = clean[idx_start:idx_start + self.len_data]

        # STFT
        noisy = rs.stft(noisy,n_fft=self.hp.audio.n_fft)
        clean = rs.stft(clean,n_fft=self.hp.audio.n_fft)

        noisy = np.abs(noisy)
        clean = np.abs(clean)
        
        noisy= torch.from_numpy(noisy)
        clean= torch.from_numpy(clean)

        noisy = torch.unsqueeze(noisy,0)
        clean = torch.unsqueeze(clean,0)

        data={}
        data["noisy_mag"] = noisy
        data["clean_mag"] = clean

        return data

    def __len__(self):
        return len(self.list_noisy)
