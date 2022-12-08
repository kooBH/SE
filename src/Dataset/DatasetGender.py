import os
from glob import glob
import torch
import librosa
import common
import numpy as np
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

def sync(x,ref):
    def cross_correlation_using_fft(x, y):
        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)

    # shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
    def compute_shift(x, y):
        assert len(x) == len(y)
        c = cross_correlation_using_fft(x, y)
        assert len(c) == len(x)
        zero_index = int(len(x) / 2) - 1
        shift = zero_index - np.argmax(c)
        return shift

    tau = compute_shift(x,ref)
    print(tau)

    if tau < 0:
        x = x[-tau:]
        shortage = len(ref) - len(x)
        y = np.pad(x,(0,shortage))
    else : 
        y = np.pad(x,(tau,0))[:-tau]
        
    # match length


    return y

class DatasetGender(torch.utils.data.Dataset):
    def __init__(self,root,hp,sr=8000,n_fft=256,req_sync=False,req_clean_spec=False):
        self.list_noisy = glob(os.path.join(root,"**","noisy","*.wav"))
        self.sr = sr
        self.n_fft = n_fft
        self.hp = hp
        self.window = torch.hann_window(n_fft)

        if hp.model.type_input == "MRI" : 
            self.func_feat = common.MRI

        self.req_sync = req_sync
        self.req_clean_spec = req_clean_spec

    def __getitem__(self, idx):
        path_noisy = self.list_noisy[idx]
        dir_item = path_noisy.split('noisy')[0]
        name_item = path_noisy.split('/')[-1]

        path_clean = os.path.join(dir_item,"clean",name_item)

        noisy_wav,_ = librosa.load(path_noisy,sr=self.sr,res_type="fft")

        # for nosie only
        try : 
            clean_wav,_ = librosa.load(path_clean,sr=self.sr,res_type="fft")
        except FileNotFoundError : 
            clean_wav = np.zeros_like(noisy_wav)

        if self.req_sync :
           noisy_wav = sync(noisy_wav,clean_wav) 

        noisy_wav = torch.from_numpy(noisy_wav)
        clean_wav = torch.from_numpy(clean_wav)

        noisy_spec =  torch.stft(
            noisy_wav,
            n_fft=self.n_fft,
            hop_length = self.hp.data.n_hop,
            return_complex=True,
            window = self.window,
            center=True
        )
        data={}

        data["input"] = self.func_feat(noisy_spec)
        data["clean_wav"] = clean_wav
        data["noisy_wav"] = noisy_wav

        if self.req_clean_spec: 
            clean_spec=  torch.stft(clean_wav,n_fft=self.hp.data.n_fft,hop_length = self.hp.data.n_hop,return_complex=True,center=True)
            clean_spec =  torch.unsqueeze(clean_spec,dim=0)
            data["clean_spec"] = clean_spec

        return data

    def __len__(self):
        return len(self.list_noisy)
