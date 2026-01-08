import torch
import torch.nn as nn
import numpy as np
import math
from multiprocessing import Pool
import os,time

class FeatureCalculator(nn.Module):
    def __init__(self, sr,n_fft, n_hop, n_erb, n_df, min_nb =2, tau=1.0,dB=True,type_window = "vorbis",normalize=True):
        super(FeatureCalculator, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.n_erb = n_erb
        self.n_df = n_df
        self.dB = dB
        self.normalize = normalize

        # STFT
        if type_window == "vorbis":
            window = self.DFN_window(n_fft)
            #wnorm = 1.0/(n_fft**2 / (2*n_hop))
            # TODO : why no wnorm on torch STFT
        elif type_window == "sine" : 
            window = torch.zeros(n_fft)
            for i in range(n_fft) :
                window[i] = torch.sin(torch.tensor(torch.pi * ((i+0.5) / n_fft)));
        else :
            raise ValueError(f"Unknown window type {type_window}")

        # torch STFT 
        #window *=wnorm
        self.register_buffer('window', window)
        
        # ERB
        erb_wd = self.cal_erb_width(sr,n_fft,n_erb,min_nb)
        self.register_buffer('erb_wd', erb_wd)
        alpha = self.compute_alpha(sr,n_hop,tau)
        self.register_buffer('alpha', torch.tensor(alpha))
        erb_fb = self.get_erb_fb(self.erb_wd, sr)
        self.register_buffer('erb_fb', erb_fb)
        erb_ifb = self.get_erb_fb(self.erb_wd, sr, inverse=True)
        self.register_buffer('erb_ifb', erb_ifb)

    def analysis(self,x):
        # x : [C,L]
        # return spec, spec: [C,T,F], ERB : [C,T,F_erb], spec_df : [C,T,F_df]

        C,L = x.shape
        self.L = L
        # [S] -> [C,S]
        # TODO : Frame by Frame
        erb_state = self.init_erb_state(self.n_erb).to(x.device)
        spec_state = self.init_spec_state(self.n_df).to(x.device)
        erb_state = erb_state.unsqueeze(0).repeat(C,1)
        spec_state = spec_state.unsqueeze(0).repeat(C,1)


        spec = torch.stft(x,n_fft=self.n_fft,hop_length = self.n_hop, window=self.window,return_complex=True)
        #spec *= self.wnorm

        # [C,T,F]
        spec = spec.permute(0,2,1)
        spec = spec[:,:-1,:]
        #erb = self.apply_erb(spec, self.erb_wd, dB=self.dB)
        #erb = self.apply_erb_mp(spec, self.erb_wd, dB=self.dB)
        erb = self.apply_erb_ver_scatter(spec, self.erb_wd, dB=self.dB)

        if self.normalize : 
            erb = self.apply_erb_norm(erb, erb_state, self.alpha)
            spec_df = self.apply_spec_norm(spec[:,:,:self.n_df].clone(),spec_state, self.alpha)
        else :
            spec_df = spec[:,:,:self.n_df].clone()

        return spec, erb, spec_df

    # Can be on GPU
    def synthesis(self,X):
        # X : [C,T,F]
        # return x : [C,:]

        X = X.permute(0,2,1)
        x = torch.istft(X,n_fft = self.n_fft,hop_length = self.n_hop, window=self.window.to(X.device), length = self.L)
        return x
    
    # Vorbis Window
    def DFN_window(self,N) : 
        window_size_h  = N/2.0
        i = np.arange(N, dtype=np.float64)
        sin_term = np.sin(0.5 * np.pi * (i + 0.5) / window_size_h)
        window = np.sin(0.5 * np.pi * sin_term**2)
        window = torch.from_numpy(window).float()
        return window

    @staticmethod
    def freq2erb(freq_hz: float) -> float:
        return 9.265 * np.log1p(freq_hz / (24.7 * 9.265))

    @staticmethod
    def erb2freq(n_erb: float) -> float:
        return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)

    @staticmethod
    def cal_erb_width(sr: int, fft_size: int, nb_bands: int, min_nb_freqs: int) -> np.ndarray:
        nyq_freq = sr / 2
        freq_width = sr / fft_size
        erb_low = FeatureCalculator.freq2erb(0.0)
        erb_high = FeatureCalculator.freq2erb(nyq_freq)

        erb = [0] * nb_bands
        step = (erb_high - erb_low) / nb_bands

        prev_freq = 0 # Last frequency band of the previous erb band
        freq_over = 0 # Number of frequency bands that are already stored in previous erb bands

        for i in range(1, nb_bands + 1):
            f = FeatureCalculator.erb2freq(erb_low + i * step)
            fb = int(round(f / freq_width))
            nb_freqs = fb - prev_freq - freq_over
            if nb_freqs < min_nb_freqs:
                # Not enough freq bins in current bark bin
                freq_over = min_nb_freqs - nb_freqs
                nb_freqs = min_nb_freqs
            else:
                freq_over = 0
            erb[i - 1] = nb_freqs
            prev_freq = fb

        erb[-1] += 1  # since we have WINDOW_SIZE/2+1 frequency bins
        total_bins = sum(erb)
        target_bins = fft_size // 2 + 1

        if total_bins > target_bins:
            erb[-1] -= (total_bins - target_bins)

        assert sum(erb) == target_bins, f"Total bins ({sum(erb)}) != {target_bins}"
        return torch.from_numpy(np.array(erb)).float()
    
    @staticmethod
    def apply_erb_per_channel(args):
        b,x,erb_width = args
        # x : [T,F]
        out = np.zeros((x.shape[0],len(erb_width)))
        for t in range(x.shape[0]) :
            bcsum = 0
            for e,band_size in enumerate(erb_width):
                for j in range(int(band_size)):
                    idx = int(bcsum + j)
                    out[t,e] += (x[t, idx].real * x[t,idx].real + x[t,idx].imag * x[t,idx].imag)
                out[t,e] /=band_size
                bcsum += band_size

        return out
    
    def apply_erb_mp(self, x: torch.tensor, erb_width : torch.tensor, dB = True) -> torch.tensor:
        # x : [C,T,F]
        out = torch.zeros((x.shape[0],x.shape[1],len(erb_width))).to(x.device)
        args = [(b,x[b].numpy(),erb_width) for b in range(x.shape[0])]
        with Pool(processes=os.cpu_count()//2) as pool:
            results = pool.map(FeatureCalculator.apply_erb_per_channel, args)
        for b,out_local in enumerate(results):
            out[b] = torch.from_numpy(out_local).float()
        if dB : 
            out = torch.log10(out + 1e-10) * 10.0
        return out
    
    def apply_erb_ver_map(self, x : torch.tensor, erb_width : torch.tensor, dB = True) -> torch.tensor:
        # x : [C,T,F]
        band_map = torch.zeros(x.shape[2], dtype=torch.long)
        bcsum = 0
        for i, size in enumerate(erb_width):
            size = int(size)
            band_map[bcsum : bcsum + size] = i
            bcsum += size
        
        x_power = x.real**2 + x.imag**2  # [C, T, F]
        
        # [C, T, F] → [C, T, B]
        out = torch.zeros(x.shape[0], x.shape[1], len(erb_width), device=x.device)
        for b in range(len(erb_width)):
            mask = (band_map == b)
            out[..., b] = (x_power[..., mask].sum(-1)) / erb_width[b]
        return out

    def apply_erb_ver_loop(self, x : torch.tensor, erb_width : torch.tensor, dB = True) -> torch.tensor:
        # x : [C,T,F]
        out = torch.zeros((x.shape[0],x.shape[1],len(erb_width))).to(x.device)

        for c in range(x.shape[0]) : 
            for t in range(x.shape[1]) : 
                bcsum = 0
                for i, band_size in enumerate(erb_width):
                    k = 1.0 / band_size
                    for j in range(int(band_size)):
                        idx = int(bcsum + j)
                        out[c,t,i] += (x[c,t,idx].real * x[c,t,idx].real + x[c,t,idx].imag * x[c,t,idx].imag) * k
                    bcsum += band_size
        if dB : 
            out = torch.log10(out + 1e-10) * 10.0
        return out
    
    def apply_erb_ver_scatter(self, x : torch.tensor, erb_width : torch.tensor, dB = True) -> torch.tensor:
        C,T,F = x.shape
        B =len(erb_width)

        band_map = torch.zeros(F, dtype=torch.long,device=x.device)
        bcsum = 0
        for i, size in enumerate(erb_width):
            size = int(size)
            band_map[bcsum : bcsum + size] = i
            bcsum += size
        
        power = (x.real ** 2 + x.imag ** 2).to(torch.float32)  # [C, T, F]
        
        power_flat = power.reshape(-1, F)  # [(C*T), F]
        band_map_exp = band_map.unsqueeze(0).expand(power_flat.shape)  # [(C*T), F]
        
        out = torch.zeros(power.shape[0] * power.shape[1], B, device=x.device)  # [(C*T), B]
        out.scatter_add_(1, band_map_exp, power_flat)  
        erb_width_float = erb_width.to(power.dtype).to(x.device)

        out = out / erb_width_float.unsqueeze(0)  # broadcasting
        
        out = out.view(C, T, B)

        if dB : 
            out = torch.log10(out + 1e-10) * 10.0
        
        return out

    @staticmethod
    def compute_alpha(sr: int, hop_size: int, tau: float = 1.0) -> float:
        dt = hop_size / sr
        alpha = math.exp(-dt / tau)
        precision = 3
        a = 1.0
        while a >= 1.0:
            a = round(alpha, precision)
            precision += 1
        return a
    
    @staticmethod
    def init_erb_state(n_erb):
        min_val = -60.0
        max_val = -90.0
        mean_norm_state = np.linspace(min_val, max_val, n_erb)
        return torch.from_numpy(mean_norm_state).float()

    @staticmethod
    def apply_erb_norm(x,state, alpha) :
        # x     : [B, T, F]
        # state : [B,F]
        B, T, F = x.shape
        #print(f"erb norm {x.shape} {state.shape}")
        for t in range(T):
            x_t = x[:, t, :]
            # Exponentially weighted moving state
            state = (1 - alpha) * x_t + alpha * state
            x[:, t, :] = (x_t - state) / 40.0
        return x
        
    @staticmethod
    def init_spec_state(n_spec) :
        min_val = 1e-3
        max_val = 1e-4
        mean_norm_state = np.linspace(min_val, max_val, n_spec)
        return torch.from_numpy(mean_norm_state).float()
        
    @staticmethod
    def apply_spec_norm(x,state,alpha) : 
        # x: [B, T, F]
        # state : [B,F]
        B, T, F = x.shape
        for t in range(T):
            norms = torch.abs(x[:,t,:])
            state = norms * (1.0 - alpha) + state * alpha
            x[:,t,:] /= torch.sqrt(state)
        return x
    
    def get_erb_fb(self, erb_wd,  sr : int,normalized : bool = True, inverse : bool = False) : 
        n_freqs = int(torch.sum(erb_wd))
        all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

        #b_pts = np.cumsum([0] + erb_wd.tolist()).astype(int)[:-1]
        b_pts = torch.cumsum(torch.cat([torch.tensor([0], dtype=erb_wd.dtype), erb_wd]), dim=0)[:-1].to(dtype=torch.int32)

        fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
        for i, (b, w) in enumerate(zip(b_pts.tolist(), erb_wd.tolist())):
            fb[b : b + int(w), i] = 1
        # Normalize to constant energy per resulting band
        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= fb.sum(dim=1, keepdim=True)
        else:
            if normalized:
                fb /= fb.sum(dim=0)
        return fb
