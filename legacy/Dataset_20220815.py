import os
import glob
import torch
import librosa

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.list_noisy = glob.glob(os.path.join(root,"noisy","*.wav"))

    def __getitem__(self, index):
        path_noisy = self.list_noisy[index]
        name_noisy = path_noisy.split('/')[-1]
        path_clean = os.path.join(self.root,"clean",name_noisy)


        noisy_wav, _ = librosa.load(path_noisy,sr=8000,mono=True)
        clean_wav , _ = librosa.load(path_clean,sr=8000,mono=True)

        noisy_wav = torch.from_numpy(noisy_wav)
        clean_wav = torch.from_numpy(clean_wav)

        noisy_spec =  torch.stft(noisy_wav,n_fft=512,return_complex=True,center=True)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)

        noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
        noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

        data={}
        data["noisy_mag"]=noisy_mag
        data["noisy_phase"]=noisy_phase
        data["clean_wav"]=clean_wav
        data["noisy_wav"] = noisy_wav
    

        return data

    def __len__(self):
        return len(self.list_noisy)


