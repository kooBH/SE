from sympy import intervals

import numpy as np
from scipy import signal
import torch
import librosa as rs

def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar

def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 10) / (rms + eps)
    y *= scalar
    return y, rms, scalar


spec_window = None

def spec_augment(x, n_fft,n_hop, window,time_width,freq_width) :

    if window is None :
        if spec_window is None :
            spec_window = torch.hann_window(n_fft)
        window = spec_window

    hop_length = n_hop

    X = torch.stft(torch.from_numpy(x),n_fft=n_fft,hop_length=hop_length,window=window,return_complex=True)

    t_width = time_width
    f_width = freq_width

    if type(t_width) is list : 
        t_width = np.random.randint(t_width[0],t_width[1])
    if type(f_width) is list : 
        f_width = np.random.randint(f_width[0],f_width[1])

    t_beg= np.random.randint( X.shape[0] - t_width)
    f_beg= np.random.randint( X.shape[1] - f_width)

    X[t_beg:t_beg+t_width,f_beg:f_beg+f_width] = 0

    y = torch.istft(X,n_fft=n_fft,hop_length=hop_length,window=window,length=len(x))

    return y.numpy()

def pre_emphasis(signal, coeff=0.99):
    # Standard pre-emphasis filter to boost high frequencies
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def adjusted_rms(signal, vad=False):
    if vad:
        intervals = rs.effects.split(signal, top_db=30) 
        signal = np.concatenate([signal[start:end] for start, end in intervals])
    signal = pre_emphasis(signal)

    signal_rms = (signal** 2).mean() ** 0.5
    return signal_rms


def mix(clean,noise,rir=None,
        # scale
        target_dB_FS = -15, 
        target_dB_FS_floating_value = 10,
        scale_range = [0.1,1.0],
        scale_method = "dB",
        # SNR
        range_SNR=[-5,25],
        # Reverb
        deverb_clean=False,
        clean_rir_len=25,
        # SpecAugmentation
        use_spec_augmentation = False,
        spec_n_fft=512,
        spec_n_hop=128,
        spec_window = None,
        spec_time_width = [10,50],
        spec_freq_width = [5,20],
        # Misc
        eps=1e-7):
    # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
    noisy_target_dB_FS = np.random.randint(
        target_dB_FS - target_dB_FS_floating_value,
        target_dB_FS + target_dB_FS_floating_value
    )
    
    # skip mixing if clean is silent
    if np.sum(np.abs(clean)) < 1e-13:
        print("skip")
        noise, _, noise_scalar = tailor_dB_FS(noise, noisy_target_dB_FS)

        return noise, clean, noise

    if rir is not None:
        if rir.ndim > 1:
            rir_idx = np.random.randint(0, rir.shape[0])
            rir = rir[rir_idx, :]

        if deverb_clean :
            peak = np.argmax(rir)
            if clean_rir_len <= 0 :
                peak += clean_rir_len
            clean_peak  = signal.fftconvolve(clean,rir[:peak])[:len(clean)]
        clean = signal.fftconvolve(clean, rir)[:len(clean)]

        if deverb_clean : 
            clean = clean_peak

    ## SNR
    clean, _ = norm_amplitude(clean)
    clean_rms = adjusted_rms(clean)

    noise, _ = norm_amplitude(noise)
    noise_rms = adjusted_rms(noise,vad=False)

    if type(range_SNR) == float :
        SNR = range_SNR
    elif type(range_SNR) == int :
        SNR = range_SNR
    elif type(range_SNR) == list :
        SNR = np.random.uniform(
            range_SNR[0],range_SNR[1]
        )
    else : 
        raise ValueError(f"{type(range_SNR)} is not supported for range_SNR")

    snr_scalar = clean_rms / (10 ** (SNR / 20)) / (noise_rms + eps)
    #snr_scalar = 10**(-SNR/20)
    noise *= snr_scalar
    noisy = clean + noise

    #if rir is not None:
    #    if self.hp.data.deverb_clean : 
    #        clean = clean_peak


    ## Scaling
    if scale_method == "dB" :
        noisy, _, noisy_scalar = tailor_dB_FS(noisy, noisy_target_dB_FS)
        clean *= noisy_scalar
        noise *= noisy_scalar
    else : 
        scale = np.random.uniform(scale_range[0],scale_range[1])
        noisy,scalar = norm_amplitude(noisy)
        noisy = noisy * scale
        clean/= scalar
        clean = clean * scale


    M = max(np.max(np.abs(noisy)),np.max(np.abs(noise)),np.max(np.abs(clean))) + eps
    if M > 1.0 : 
        noisy = noisy / M
        clean = clean / M
        noise = noise / M

    if use_spec_augmentation : 
        noisy = spec_augment(noisy, spec_n_fft, spec_n_hop, spec_window, spec_time_width, spec_freq_width)

    return noisy, clean, noise


if __name__ == "__main__" : 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", help = "Clean Directory",type=str)
    parser.add_argument("--noise", help = "Noise Directory",type=str)
    parser.add_argument("--output", help = "Output Directory",type=str)
    parser.add_argument("-n", "--num", type=int, default=1000, help="Number of output samples")
    args = parser.parse_args()

    
    