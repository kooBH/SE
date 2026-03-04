import os
from os.path import join
from glob import glob
import torch
import librosa as rs
import soundfile as sf
import numpy as np
import random
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

from Dataset.Augmentation import gen_noise, rand_biquad_filter,remove_dc,rand_resample

def get_list(item,format) : 
    list_item = []
    if type(item) is str :
        list_item = glob(join(item,"**",format),recursive=True)
    elif type(item) is list :
        for i in item : 
            list_item += glob(join(i,"**",format),recursive=True)
    return list_item

"""
VoiceBank+Demand Dataset

noisy/clean data pair has same file name
"""

class DatasetVD(torch.utils.data.Dataset):
    def __init__(self,hp,is_train=True):
        self.hp = hp
        self.is_train = is_train
        if is_train : 
            self.list_noisy = get_list(hp.data.noisy,"*.wav")
        else :
            self.list_noisy = glob(join(hp.data.dev.root,"*","noisy","*.wav"),recursive=True)

        self.len_data = hp.data.len_data
        self.sr = hp.data.sr

        self.window = torch.hann_window(hp.audio.n_fft) 

        if is_train : 
            print("DatasetVD[train:{}] : {}".format(is_train,len(self.list_noisy)))
        else :
            print("DatasetVD[train:{}] : {}".format(is_train,len(self.list_noisy)))

    def match_length(self,wav,idx_start=None) : 
        if len(wav) > self.len_data : 
            left = len(wav) - self.len_data
            if idx_start is None :
                idx_start = np.random.randint(left)
            wav = wav[idx_start:idx_start+self.len_data]
        elif len(wav) < self.len_data : 
            shortage = self.len_data - len(wav) 
            wav = np.pad(wav,(0,shortage),mode="wrap")
        return wav, idx_start

    def get_clean_path(self,path_noisy):

        if self.is_train : 
            base_name = os.path.basename(path_noisy)
            path_clean = os.path.join(self.hp.data.clean,base_name)
        else : 
            path_after_root = path_noisy.split(self.hp.data.dev.root)[-1]
            dev_type = path_after_root.split("/")[0]

            fid = path_noisy.split("_")[-1]
            fid = fid.split(".")[0]

            path_clean = os.path.join(self.hp.data.dev.root,dev_type,"clean","clean_fileid_{}.wav".format(fid))

        return path_clean

    def __getitem__(self, idx):
        path_noisy = self.list_noisy[idx]
        path_clean = self.get_clean_path(path_noisy)
            
        clean = rs.load(path_clean,sr=self.sr)[0]
        noisy = rs.load(path_noisy,sr=self.sr)[0]

        clean, idx_start = self.match_length(clean)
        noisy,_ = self.match_length(noisy,idx_start)

        noisy = torch.FloatTensor(noisy)
        clean = torch.FloatTensor(clean)

        data = {"noisy":noisy,"clean":clean}
        return  data

    def __len__(self):
        return len(self.list_noisy)

    def spec_augment(self, x):

        n_fft = self.hp.audio.n_fft
        hop_length = self.hp.audio.n_hop

        X = torch.stft(torch.from_numpy(x),n_fft=n_fft,hop_length=hop_length,window=self.window,return_complex=True)

        t_width = self.hp.data.spec_augmentation.time_width
        f_width = self.hp.data.spec_augmentation.freq_width

        if type(t_width) is list : 
            t_width = np.random.randint(t_width[0],t_width[1])
        if type(f_width) is list : 
            f_width = np.random.randint(f_width[0],f_width[1])

        t_beg= np.random.randint( X.shape[0] - t_width)
        f_beg= np.random.randint( X.shape[1] - f_width)

        X[t_beg:t_beg+t_width,f_beg:f_beg+f_width] = 0

        y = torch.istft(X,n_fft=n_fft,hop_length=hop_length,window=self.window,length=len(x))

        return y.numpy()


## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("../")
    from utils.hparams import HParam
    hp = HParam("../../config/mpSEv2/v138.yaml","../../config/mpSEv2/default.yaml")

    db = DatasetVD(hp,is_train=True)

    def check(db) : 
        for i in range(10000): 
            sample_clean = db[i]["clean"]
            sample_noisy = db[i]["noisy"]

            abs_clean = np.abs(sample_clean)
            abs_noisy = np.abs(sample_noisy)

            if np.isnan(sample_clean).any() : 
                import pdb; pdb.set_trace()

            #print("shape {:.3f} {:.3f} | avg {:.3f} {:.3f} | sum {:.3f} {:.3f} | max {:.3f} {:.3f}".format(sample_clean.shape[0], sample_noisy.shape[0], np.mean(abs_clean), np.mean(abs_noisy), np.sum(abs_clean), np.sum(abs_noisy), np.max(abs_clean), np.max(abs_noisy)))

    check(db)
    import pdb; pdb.set_trace()
  




