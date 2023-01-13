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

        self.use_cdr = hp.model.use_cdr

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
            data = torch.from_numpy(data)
        else : 
            # [2, F ,T]
            data = np.stack((data.real,data.imag))
            data = torch.from_numpy(data)

        if len(data.shape) < 3 :
            data = torch.unsqueeze(data,0)

        return data

    def __getitem__(self, idx):
        ## Path Data
        path_noisy = self.list_noisy[idx]
        # it should be with out id, there was a mistake
        name_noisy = path_noisy.split("/")[-1]


        if self.use_cdr : 
            # ex : /home/data/kbh/SPEAR_seg/Train/clean/Dataset_2_Session_5_01_ch_1_seg_240000.wav
            name_clean = name_noisy
        else : 
            id_ch = np.random.randint(2)
            # Dataset_2_Session_10_00_ch_0_seg_120000.wav
            name_clean = "_".join(name_noisy.split("_")[:6])+"_"+str(id_ch)+"_"+"_".join(name_noisy.split("_")[7:])


        path_clean = os.path.join(self.hp.data.root,self.train_dev,"clean",name_clean)

        ## Load data
        noisy = rs.load(path_noisy,sr=self.hp.data.sr)[0]
        clean = rs.load(path_clean,sr=self.hp.data.sr)[0]

        # cut
        #idx_start = np.random.randint(len(noisy)-self.len_data)

        #noisy = noisy[idx_start:idx_start + self.len_data]
        #clean = clean[idx_start:idx_start + self.len_data]

        if self.hp.model.normalize : 
            noisy= noisy/(np.max(np.abs(noisy))+1e-7)
            clean= (clean+1e-7)/(np.max(np.abs(clean))+1e-7)

        # pad
        if len(noisy) < self.len_data : 
            shortage  = self.len_data - len(noisy)
            noisy = np.pad(noisy,(0,shortage))

        if len(clean) < self.len_data : 
            shortage  = self.len_data - len(clean)
            clean = np.pad(clean,(0,shortage))

        # sample 
        if len(noisy) > self.len_data :
            idx_start = np.random.randint(len(noisy)-self.len_data)
            noisy = noisy[idx_start:idx_start+self.len_data]
            clean = clean[idx_start:idx_start+self.len_data]

        # feature
        noisy= DatasetSPEAR.get_feature(noisy,self.hp)
        clean= DatasetSPEAR.get_feature(clean,self.hp)


        if self.use_cdr : 
            path_cdr = os.path.join(self.hp.data.root,self.train_dev,"cdr",name_clean)
            cdr = rs.load(path_cdr,sr=self.hp.data.sr)[0]

            if len(cdr) < self.len_data : 
                shortage  = self.len_data - len(cdr)
                cdr = np.pad(cdr,(0,shortage))
            
            if len(cdr) > self.len_data :
                idx_start = np.random.randint(len(cdr)-self.len_data)
                cdr = cdr[idx_start:idx_start+self.len_data]

            #cdr = cdr[idx_start:idx_start + self.len_data]
            if self.hp.model.normalize : 
                cdr = cdr/(np.max(np.abs(cdr))+1e-7)
            cdr = DatasetSPEAR.get_feature(cdr,self.hp)
            noisy = torch.cat((noisy,cdr),0)

        data={}
        data["noisy"] = noisy
        data["clean"] = clean

        return data

    def __len__(self):
        return len(self.list_noisy)
