import numpy as np
import scipy.fft as fft
import random
import torch
import scipy.signal as signal
import librosa as rs

"""
    `f_decay`: Decay variable. Typical values for common noises are:
        - white: `0.0`
        - pink: `1.0`
        - brown: `2.0`
        - blue: `-1.0`

    TODO : Fix pink noise bug(f_decay > 0.0 case)
"""
def gen_noise(n_sample, n_freq = 2048, f_decay = 0.0, cutoff = 20, sr = 16000):
    if type(f_decay) is list : 
        f_decay = random.uniform(f_decay[0], f_decay[1])

    T = int(n_sample / n_freq)+1

    freqs = np.fft.rfftfreq(n_freq, 1.0 / n_freq)
    freqs[0] = 1e-6

    filter = freqs ** (-f_decay / 2.0)

    # cutoff for stability
    cutoff_freq = cutoff
    cutoff_bin = int(cutoff_freq * n_freq / sr) + 1
    filter[:cutoff_bin] = filter[cutoff_bin]

    noise_complex = np.random.randn(T,len(freqs)) + 1j * np.random.randn(T,len(freqs))

    noise = fft.irfft(noise_complex, n=n_freq)
    noise = noise.flatten()

    filtered_complex = noise_complex * filter
    noise = fft.irfft(filtered_complex, n=n_freq)
    noise = noise.flatten()
    noise = noise[:n_sample]
    
    # Normalize
    max_abs_val = np.max(np.abs(noise))
    noise = noise / max_abs_val

    return noise

"""
Biquad Filter 
https://www.w3.org/TR/audio-eq-cookbook/#formulae

"""
def high_shelf(center_freq, gain_db, q_factor, sr):
    w0 = 2. * np.pi * center_freq / sr
    amp = 10.**(gain_db / 40.)
    alpha = np.sin(w0) / (2. * q_factor)
    
    b0 = amp * ((amp + 1.) + (amp - 1.) * np.cos(w0) + 2. * np.sqrt(amp) * alpha)
    b1 = -2. * amp * ((amp - 1.) + (amp + 1.) * np.cos(w0))
    b2 = amp * ((amp + 1.) + (amp - 1.) * np.cos(w0) - 2. * np.sqrt(amp) * alpha)
    a0 = (amp + 1.) - (amp - 1.) * np.cos(w0) + 2. * np.sqrt(amp) * alpha
    a1 = 2. * ((amp - 1.) - (amp + 1.) * np.cos(w0))
    a2 = (amp + 1.) - (amp - 1.) * np.cos(w0) - 2. * np.sqrt(amp) * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def high_pass(center_freq, q_factor, sr):
    w0 = 2. * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2. * q_factor)

    b0 = (1. + np.cos(w0)) / 2.
    b1 = -(1. + np.cos(w0))
    b2 = (1. + np.cos(w0)) / 2.
    a0 = 1. + alpha
    a1 = -2. * np.cos(w0)
    a2 = 1. - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def low_shelf(center_freq, gain_db, q_factor, sr):
    w0 = 2. * np.pi * center_freq / sr
    amp = 10.**(gain_db / 40.)
    alpha = np.sin(w0) / (2. * q_factor)
    
    b0 = amp * ((amp + 1.) - (amp - 1.) * np.cos(w0) + 2. * np.sqrt(amp) * alpha)
    b1 = 2. * amp * ((amp - 1.) - (amp + 1.) * np.cos(w0))
    b2 = amp * ((amp + 1.) - (amp - 1.) * np.cos(w0) - 2. * np.sqrt(amp) * alpha)
    a0 = (amp + 1.) + (amp - 1.) * np.cos(w0) + 2. * np.sqrt(amp) * alpha
    a1 = -2. * ((amp - 1.) + (amp + 1.) * np.cos(w0))
    a2 = (amp + 1.) + (amp - 1.) * np.cos(w0) - 2. * np.sqrt(amp) * alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def low_pass(center_freq, q_factor, sr):
    w0 = 2. * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2. * q_factor)
    
    b0 = (1. - np.cos(w0)) / 2.
    b1 = 1. - np.cos(w0)
    b2 = b0
    a0 = 1. + alpha
    a1 = -2. * np.cos(w0)
    a2 = 1. - alpha
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def peaking_eq(center_freq, gain_db, q_factor, sr):
    w0 = 2. * np.pi * center_freq / sr
    amp = 10.**(gain_db / 40.)
    alpha = np.sin(w0) / (2. * q_factor)
    
    b0 = 1. + alpha * amp
    b1 = -2. * np.cos(w0)
    b2 = 1. - alpha * amp
    a0 = 1. + alpha / amp
    a1 = -2. * np.cos(w0)
    a2 = 1. - alpha / amp

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def notch(center_freq, q_factor, sr):
    w0 = 2. * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2. * q_factor)
    
    b0 = 1.
    b1 = -2. * np.cos(w0)
    b2 = 1.
    a0 = 1. + alpha
    a1 = -2. * np.cos(w0)
    a2 = 1. - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def rand_biquad_filter(x,
    sr=16000,
    max_iter= 3,
    gain_db_high= 15.,
    gain_db_low = -15.,
    q_low= 0.5,
    q_high= 1.5,
): 
    n_iter = np.random.randint(1, max_iter+1)
    for _ in range(n_iter):
        filter_type = np.random.choice(
            ['high_shelf', 'high_pass', 'low_shelf', 'low_pass', 'peaking_eq', 'notch']
        )
        gain_db = np.random.uniform(gain_db_low, gain_db_high)
        q_factor = np.random.uniform(q_low, q_high)
        if filter_type == 'low_pass':
            f_high = 8000.
            f_low = 4000.
            f0 = np.random.uniform(f_low, f_high)
            b, a = low_pass(f0, q_factor, sr)
        elif filter_type == 'high_shelf':
            f_high = 8000.
            f_low = 1000.
            f0 = np.random.uniform(f_low, f_high)
            b, a = high_shelf(f0, gain_db, q_factor, sr)
        elif filter_type == 'high_pass':
            f_high = 400.
            f_low = 40.
            f0 = np.random.uniform(f_low, f_high)
            b, a = high_pass(f0, q_factor, sr)
        elif filter_type == 'low_shelf':
            f_high = 1000.
            f_low = 40.
            f0 = np.random.uniform(f_low, f_high)
            b, a = low_shelf(f0, gain_db, q_factor, sr)
        elif filter_type == 'peaking_eq':
            f_high = 4000.
            f_low = 40.
            f0 = np.random.uniform(f_low, f_high)
            b, a = peaking_eq(f0, gain_db, q_factor, sr)
        elif filter_type == 'notch':
            f_high = 4000.
            f_low = 40.
            f0 = np.random.uniform(f_low, f_high)
            b, a = notch(f0, q_factor, sr)
        else : 
            raise ValueError("Unknown filter type")
        
        # Apply Filter
        x = signal.lfilter(b, a, x)
    
    return x


def remove_dc(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    x = x - mean
    return x

def rand_resample(x, sr=16000, r_low=0.9, r_high = 1.1) :
    # x : [L]
    sr_hat = np.random.uniform(r_low, r_high)*sr
    sr_hat = int(np.round(sr_hat/500.0)*500.0)

    if sr_hat == sr :
        return x
    
    len_x = x.shape[-1]
    x_hat = rs.resample(x, orig_sr=sr, target_sr = sr_hat)
    x_hat = x_hat[:len_x]

    return x_hat


def rand_clipping(x, c_min = 0.01, c_max = 0.25):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return x

    c = np.random.uniform(c_min * max_val, c_max * max_val)

    return np.clip(x, -c, c)