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

    def get_feature(self,wav):
        stft = torch.stft(audio, frame_size, hop_size, frame_size, window, return_complex=True)
        mag = torch.abs(stft)
        # In the paper, they use real and imag for phase, e**(1j*phi), but it's not needed
        phase = torch.angle(stft)
        pcen = torch.Tensor(librosa.pcen(mag.cpu().detach().numpy()**2, sr=fs, hop_length=hop_size, axis=-2)).to(stft.device)
        return torch.stack([mag, phase, pcen], dim=-2)

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

    def mix(self,s):
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def __getitem__(self, idx):
        ## Path Data
        path_noisy = self.list_noisy[idx]
        # it should be with out id, there was a mistake
        name_noisy = path_noisy.split("/")[-1]

        return data

    def __len__(self):
        return len(self.list_noisy)



## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("./")
    from utils.hparams import HParam
    hp = HParam("../config/SPEAR/v20.yaml","../config/SPEAR/default.yaml")
  




