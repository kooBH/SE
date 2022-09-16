import os
from glob import glob
import torch
import librosa


class DatasetGender(torch.utils.data.Dataset):
    def __init__(self,root,hp,sr=8000,n_fft=256):
        self.list_noisy = glob(os.path.join(root,"**","noisy","*.wav"))
        self.sr = sr
        self.n_fft = n_fft
        self.hp = hp

    def __getitem__(self, idx):
        path_noisy = self.list_noisy[idx]
        dir_item = path_noisy.split('noisy')[0]
        name_item = path_noisy.split('/')[-1]

        path_clean = os.path.join(dir_item,"clean",name_item)

        noisy_wav,_ = librosa.load(path_noisy,sr=self.sr)
        clean_wav,_ = librosa.load(path_clean,sr=self.sr)

        noisy_wav = torch.from_numpy(noisy_wav)
        clean_wav = torch.from_numpy(clean_wav)

        noisy_spec =  torch.stft(noisy_wav,n_fft=self.n_fft,return_complex=True,center=True)

        data={}
        if self.hp.model.mag_only : 
            noisy_mag = torch.abs(noisy_spec)
            noisy_phase = torch.angle(noisy_spec)

            noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
            noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

            data["input"]=noisy_mag
            data["noisy_phase"]=noisy_phase
        else :
            noisy_spec = torch.permute(torch.view_as_real(noisy_spec),(2,0,1))
            data["input"]=noisy_spec

        data["clean_wav"]=clean_wav
        data["noisy_wav"] = noisy_wav

        if self.hp.loss.type == "mwMSELoss" : 
            clean_spec=  torch.stft(clean_wav,n_fft=self.n_fft,return_complex=True,center=True)
            clean_spec =  torch.unsqueeze(clean_spec,dim=0)
            data["clean_spec"] = clean_spec

        return data

    def __len__(self):
        return len(self.list_noisy)


