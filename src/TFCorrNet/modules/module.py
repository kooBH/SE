import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn


from utils.decorators import *
from .network import *




class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True
        self.beta.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class ISCMembedding(nn.Module):
    def __init__(self, beta_type: str, d_freq: int, d_model: int, num_mics: int):
        super().__init__()
        if beta_type == 'vector':
            print('vector')
            self.exponent = nn.Parameter(torch.zeros(d_freq,1), requires_grad=True)
        elif beta_type == 'scalar':
            print('scalar')
            self.exponent = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        elif beta_type == 'fixed_mag':
            print('fixed to 0.5')
            self.exponent = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        elif beta_type == 'fixed_no':
            print('no')
            self.exponent = nn.Parameter(torch.tensor(-1.0e10), requires_grad=False)

        self.IPD_factor = nn.Parameter(torch.zeros(d_freq,1), requires_grad=True)
        self.sigmoid  = torch.nn.Sigmoid()
        self.embed = nn.Conv1d((num_mics)*(num_mics+1), d_model, 5, padding='same')
        self.layer_norm = nn.LayerNorm([d_model, d_freq])
        
    def forward(self, x: torch.tensor):
        # x : B, T, d_freq, M

        B, T, d_freq, M = x.shape
        x_r = x[...,:M//2] # [B, T, 257, M]
        x_i = x[...,M//2:] # [B, T, 257, M]

        # x_mag = torch.sqrt(torch.square(x_r) + torch.square(x_i))
        xs = self.get_scm(x_r.permute(0, 3, 2, 1), x_i.permute(0, 3, 2, 1)) # B, T, F, M*M
        xs_abs = xs.abs()
        beta_phat = torch.pow(xs_abs,self.sigmoid(self.exponent))
        xs_abs = xs_abs / (beta_phat + 1.0e-10)
        xs_angle = xs.angle() * self.sigmoid(self.IPD_factor)
        # print(self.sigmoid(self.IPD_factor))
        
        xs = torch.polar(xs_abs, xs_angle)
        xs = torch.view_as_real(xs)
        B, T, F, C, _ = xs.shape
        xs = xs.view(B, T, F, C*2).contiguous()

        B, T, F, C = xs.shape
        xs = xs.permute(0, 2, 3, 1).contiguous().view(B*F, C, T)
        xs = self.embed(xs)
        C = xs.shape[-2]
        xs = xs.view(B, F, C, T).contiguous()
        xs = xs.permute(0, 3, 2, 1).contiguous() # B, T, C, F
        xs = self.layer_norm(xs)

        return xs

    def get_scm(self,
                x_r, # [... , ch , freq , time]
                x_i, # [... , ch , freq , time]
                mask=None # [... , freq , time]
                ): # [... , freq , ch*ch]

        """Compute weighted Spatial Covariance Matrix."""
        
        x = torch.complex(x_r,x_i)

        # if mask is not None:
        #     x = x * torch.unsqueeze(torch.clamp(mask,min=1.0e-4),-3)

        x = x - torch.mean(x, dim=-1, keepdim=True)
        # pre-emphasis / normalize
        x_abs = x.abs()
        x_mean = torch.mean(x_abs**2, dim=-1, keepdim=True)
        x_norm = torch.sqrt(torch.clamp(torch.sum(x_mean,dim=1, keepdim=True),min=1.0e-10))
        x = x / x_norm

        x = torch.transpose(x, -3, -2)  # shape (batch, freq, ch, time)
        # outer product:
        # (..., ch_1, time) x (..., ch_2, time) -> (..., time, ch_1, ch_2)    
        scm = torch.einsum("...ct,...et->...tce", [x, x.conj()])
        B, F, T, C1, C2 = scm.shape
        idx = torch.triu_indices(C1,C2)
        scm = scm[...,idx[0],idx[1]]
        B, F, T, C = scm.shape

        scm = scm.permute(0, 2, 1, 3) # [B, T, F, C]
        return scm        
    


class Multi_Path_Block(nn.Module):
    def __init__(self, channel_module: dict,  spectral_module: dict, low_rank_module: dict):
        """Construct an EncoderLayer object."""
        super(Multi_Path_Block, self).__init__()

        class LowRankModule(nn.Module):
            def __init__(self, d_model: int, d_proj: int, d_freq: int, d_embed: int, n_head: int, kernel_size: int, dropout_rate: float):
                super().__init__()
                
                self.embed_in = nn.Sequential(nn.Linear(d_model, 4*d_proj),
                                              nn.GLU(dim=-1),
                                              nn.Linear(2*d_proj, d_proj))
                self.freq_embed = nn.Linear(d_freq, d_embed)
                self.block = GLformer(d_embed, n_head, kernel_size, dropout_rate)
                self.freq_out = nn.Linear(d_embed, d_freq)
                self.embed_out = nn.Linear(d_proj, d_model)

            def forward(self, x, pos_k):
                '''
                input : B, T, C, F
                output : B, T, C, F
                '''
                input = x
                B, T, C, F = x.shape
                x = x.permute(0, 1, 3, 2).view(B*T, F, C).contiguous() # B*T, F, C
                x = self.embed_in(x) # B*T, F, C'
                C_ = x.shape[-1]
                x = x.view(B, T, F, C_).permute(0,3,1,2).contiguous() # B, C, T, F
                x = x.view(B*C_, T, F).contiguous()
                x = self.freq_embed(x) # B*C, T, G
                x = self.block(x, pos_k) # B*C, T, G
                x = self.freq_out(x) # B*C, T, F
                x = x.view(B, C_, T, F).permute(0, 2, 3, 1).contiguous() # B, T, F, C
                x = x.view(B*T, F, C_).contiguous()
                x = self.embed_out(x) # B*T, F, C
                C = x.shape[-1]
                x = x.view(B, T, F, C).permute(0, 1, 3, 2).contiguous() # B, T, C, F
                
                return input + x


        class SpectralModule(nn.Module):
            
            def __init__(self, d_model: int, n_head: int, kernel_size: int, dropout_rate: float):
                super().__init__()
                
                self.block = GLformer(d_model, n_head, kernel_size, dropout_rate)

            def forward(self, x: torch.tensor, pos_k: torch.tensor):
                '''
                input : B, T, C, F
                output : B, T, C, F
                '''
                
                B, T, C, F = x.shape
                x = x.permute(0, 1, 3, 2).contiguous() #* B, T, F, C
                x = x.view(B*T, F, C).contiguous()
                x = self.block(x, pos_k)
                x = x.view(B, T, F, C).contiguous()
                x = x.permute(0, 1, 3, 2).contiguous() #* B, T, C, F
                
                return x

        class ChannelModule(nn.Module):
            def __init__(self, d_model: int, n_head: int, kernel_size: int, dropout_rate: float):
                super().__init__()
                
                self.block = GLformer(d_model, n_head, kernel_size, dropout_rate)

            def forward(self, x: torch.tensor, pos_k: torch.tensor):
                '''
                input : B, T, C, F
                output : B, T, C, F
                '''
                
                B, T, C, F = x.shape
                x = x.permute(0, 3, 1, 2).contiguous() #* B, F, T, C
                x = x.view(B*F, T, C).contiguous()
                x = self.block(x, pos_k)
                x = x.view(B, F, T, C).contiguous()
                x = x.permute(0, 2, 3, 1).contiguous() #* B, T, C, F
                
                return x

        self.frame_wise_block = SpectralModule(**spectral_module)
        self.freq_wise_block = ChannelModule(**channel_module)
        self.channel_wise_block_1 = LowRankModule(**low_rank_module)
        self.channel_wise_block_2 = LowRankModule(**low_rank_module)


    def forward(self, x: torch.tensor, pos_k: list):
        # pos_k : list of torch.tensor

        x = self.frame_wise_block(x, pos_k[1])
        x = self.channel_wise_block_1(x, pos_k[0])
        x = self.freq_wise_block(x, pos_k[2])
        x = self.channel_wise_block_2(x, pos_k[0])
        
        return x




class MaskEstimator(nn.Module):
    def __init__(self, d_model, num_mics, frame_win_len):
        super().__init__()

        class MagMaskEstim(nn.Module):
            def __init__(self, d_model, num_mics, frame_win_len):
                super().__init__()
                pad_len = (frame_win_len-1)//2
                self.conv = nn.Conv2d(d_model, frame_win_len*num_mics, (frame_win_len,1), padding=(pad_len,0))
                self.act = nn.Softplus()
            
            def forward(self, x):
                x = self.conv(x)
                mag_mask = self.act(x)
                return mag_mask

        class PhaseMaskEstim(nn.Module):
            def __init__(self, d_model, num_mics, frame_win_len):
                super().__init__()
                pad_len = (frame_win_len-1)//2
                self.conv_real = nn.Conv2d(d_model, frame_win_len*num_mics, (frame_win_len,1), padding=(pad_len,0))
                self.conv_imag = nn.Conv2d(d_model, frame_win_len*num_mics, (frame_win_len,1), padding=(pad_len,0))
                self.eps = 1.0e-10
                
            def forward(self, x):
                x_real = self.conv_real(x)
                x_imag = self.conv_imag(x)
                pha_mask = torch.atan2(x_imag+self.eps, x_real+self.eps)
                return pha_mask

        #self.num_mics = num_mics+1
        self.num_mics = num_mics
        self.magmask = MagMaskEstim(d_model, self.num_mics, frame_win_len)
        self.phasemask = PhaseMaskEstim(d_model, self.num_mics, frame_win_len)
        self.frame_win_len = frame_win_len

        
    def forward(self, x):
        # x : B, T, C, F
         
        M = self.num_mics
        W = self.frame_win_len         

        B, T, C, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # B, C, T, F
        mag_mask = self.magmask(x) # B, W, T, F
        phase_mask = self.phasemask(x) # B, W, T, F
        mask_real = mag_mask * torch.cos(phase_mask) 
        mask_imag = mag_mask * torch.sin(phase_mask)
        mask = torch.stack([mask_real, mask_imag], dim=-1) # B, M*W, T, F, 2
        mask = mask.view(B, M, W, T, F, 2)
        mask = mask.permute(0, 1, 2, 4, 3, 5).contiguous()  # B, M, W, F, T, 2

        return mask



class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, n_head, maxlen=8000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model//n_head)
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model//n_head)
        self.embed_v = embed_v

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None