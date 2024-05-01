import torch
import numpy as np
import librosa as rs

# TODO : window
def feat_TRUNet(audio,n_fft=512,n_hop=128,sr=16000) : 
    stft = torch.stft(audio, n_fft, n_hop, n_fft, window, return_complex=True)
    mag = torch.abs(stft)
    # In the paper, they use real and imag for phase, e**(1j*phi), but it's not needed
    phase = torch.angle(stft)
    pcen = torch.Tensor(rs.pcen(mag.cpu().detach().numpy()**2, sr=sr, hop_length=n_hop, axis=-2)).to(stft.device)
    return torch.stack([mag, phase, pcen], dim=-2)