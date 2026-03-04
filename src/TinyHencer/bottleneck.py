import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import math
from torch import Tensor
from dataclasses import dataclass
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from torch.nn.utils.parametrize import remove_parametrizations

def calculate_positional_embedding(channels: int, freq: int) -> Tensor:
    # f0: [1/F, 2/F, ..., 1] * pi
    # c: [1, ..., F-1] -> log-spaced, numel = C//2
    f = torch.arange(1, freq+1, dtype=torch.float32) * (math.pi / freq)
    c = torch.linspace(
        start=math.log(1),
        end=math.log(freq-1),
        steps=channels//2,
        dtype=torch.float32
    ).exp()
    grid = f.view(-1, 1) * c.view(1, -1)            # [F, C//2]
    pe = torch.cat((grid.sin(), grid.cos()), dim=1) # [F, C]
    return pe

class ChannelsLastBatchNorm(nn.BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [T, B, F, C]"""
        T, B, F, C = x.shape
        x = x.view(T*B*F, C, 1)
        return super().forward(x).view(T, B, F, C)


class ChannelsLastSyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [T, B, F, C]"""
        T, B, F, C = x.shape
        x = x.view(T*B*F, C, 1)
        return super().forward(x).view(T, B, F, C)


class Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        attn_bias: bool,
    ):
        super().__init__()
        self.channels = channels // num_heads
        self.num_heads = num_heads
        self.scale: float = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels*3, bias=attn_bias)

    def forward(self, x: Tensor) -> Tensor:
        '''input / output: [T*B, F, C]'''
        TB, Freq, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(TB, Freq, self.num_heads, -1).transpose(1, 2)     # [TB, NH, F, C']
        q = qkv[:, :, :, :self.channels]
        k = qkv[:, :, :, self.channels:self.channels*2]
        v = qkv[:, :, :, self.channels*2:]
        out = F.scaled_dot_product_attention(q, k, v, scale=None)     # [TB, NH, F, C'']
        out = out.transpose(1, 2).reshape(TB, Freq, -1)     # [TB, F, Cout]
        return out


class RNNFormerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        freq: int,
        num_heads: int,
        weight_norm: bool,
        activation: str,
        activation_kwargs: tp.Dict[str, tp.Any],
        positional_embedding: tp.Optional[str],
        attn_bias: bool = False,
        eps: float = 1e-8,
        post_act: bool = False,
        pre_norm: bool = False,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.freq = freq
        self.pre_norm = pre_norm

        def Act(**kwargs):
            if post_act:
                return getattr(nn, activation)(**kwargs)
            return nn.Identity(**kwargs)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = ChannelsLastSyncBatchNorm
        else:
            BatchNorm = ChannelsLastBatchNorm

        self.rnn_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.rnn = nn.GRU(channels, channels, batch_first=False)
        self.rnn_fc = nn.Linear(channels, channels, bias=False)
        self.rnn_post_norm = BatchNorm(channels, eps)
        self.rnn_act = Act(**activation_kwargs)

        self.attn_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.attn = Attention(channels, num_heads, attn_bias)
        self.attn_fc = nn.Linear(channels, channels, bias=False)
        self.attn_post_norm = BatchNorm(channels, eps)
        self.attn_act = Act(**activation_kwargs)

        self.dropout = nn.Identity() if p_dropout == 0.0 else nn.Dropout(p=p_dropout, inplace=True)

        self.pe = None
        if positional_embedding is not None:
            pe = calculate_positional_embedding(channels, freq) # [F, C]
            if positional_embedding == "fixed":
                self.register_buffer("pe", pe)
                self.pe: Tensor
            elif positional_embedding == "train":
                self.pe = nn.Parameter(pe)

        self.weight_norm = weight_norm
        if weight_norm:
            self.rnn = weight_norm_fn(self.rnn, name="weight_ih_l0")
            self.rnn = weight_norm_fn(self.rnn, name="weight_hh_l0")
            self.attn.qkv = weight_norm_fn(self.attn.qkv)

    def remove_weight_reparameterizations(self):
        if self.weight_norm:
            remove_parametrizations(self.rnn, "weight_ih_l0")
            remove_parametrizations(self.rnn, "weight_hh_l0")
            remove_parametrizations(self.attn.qkv, "weight")
            self.flatten_parameters()
            self.weight_norm = False

        for fc, norm in (
            (self.rnn_fc, self.rnn_post_norm),
            (self.attn_fc, self.attn_post_norm)
        ):
            std = norm.running_var.add(norm.eps).sqrt()
            fc.weight.data *= norm.weight.view(-1, 1) / std.view(-1, 1)
            fc.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
        self.rnn_post_norm = nn.Identity()
        self.attn_post_norm = nn.Identity()

        if self.pre_norm:
            # w @ (x - mean) / std + bias
            # = (w \cdot gamma) @ x + (bias + w @ beta)
            # where gamma = 1/std, beta = -mean/std
            # 1. Attn
            norm = self.attn_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.attn.qkv.weight.data @ beta.view(-1, 1)).squeeze(1)

            self.attn.qkv.weight.data /= std
            attn_bias = torch.zeros(self.attn.qkv.weight.size(0))
            if self.attn.qkv.bias is not None:
                attn_bias = self.attn.qkv.bias.data
            self.attn.qkv.bias = nn.Parameter(attn_bias + w_matmul_beta)
            self.attn_pre_norm = nn.Identity()

            # 2. GRU
            norm = self.rnn_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.rnn.weight_ih_l0.data @ beta.view(-1, 1)).squeeze(1)

            self.rnn.weight_ih_l0.data /= std
            self.rnn.bias_ih_l0.data.add_(w_matmul_beta)
            self.rnn_pre_norm = nn.Identity()

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> Tensor:
        return x.new_zeros(1, self.freq, self.rnn.hidden_size)

    def forward(self, x: Tensor, h: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        TIME, BATCH, FREQ, CH = x.shape     # [T, B, F, C]
        x_in = x
        x = self.rnn_pre_norm(x)            # [T, B, F, C]
        x = x.view(TIME, FREQ*BATCH, CH)    # [T, B*F, C]
        x, h = self.rnn(x, h)               # [T, B*F, C]
        x = x.view(TIME, BATCH, FREQ, CH)   # [T, B, F, C]
        x = self.rnn_fc(x)                  # [T, B, F, C]
        x = self.dropout(x)
        x = self.rnn_post_norm(x)           # [T, B, F, C]
        x = self.rnn_act(x)                 # [T, B, F, C]
        x = x.add_(x_in)                    # [T, B, F, C]

        if self.pe is not None:
            x = x.add_(self.pe)
        x_in = x
        x = self.attn_pre_norm(x)           # [T, B, F, C]
        x = x.view(TIME*BATCH, FREQ, CH)    # [T*B, F, C]
        x = self.attn(x)                    # [T*B, F, C]
        x = x.view(TIME, BATCH, FREQ, CH)   # [T, B, F, C]
        x = self.attn_fc(x)                 # [T, B, F, C]
        x = self.dropout(x)
        x = self.attn_post_norm(x)          # [T, B, F, C]
        x = self.attn_act(x)                # [T, B, F, C]
        x = x.add_(x_in)                    # [T, B, F, C
        return x, h


@dataclass
class RNNFormerConfig:
    num_blocks: int = 3
    channels: int = 32
    freq: int = 32
    num_heads: int = 4
    eps: float = 1e-8
    positional_embedding: tp.Optional[str] = "train"    # None | "fixed" | "train"
    attn_bias: bool = False
    post_act: bool = False
    pre_norm: bool = False
    p_dropout: float = 0.0


def rf_pre_post_lin(
    freq: int,
    n_filter: int,
    init: tp.Optional[str],
    bias: bool,
    sr: int = 16_000
) -> tp.Tuple[nn.Module, nn.Module]:
    assert init in [None, "linear", "linear_fixed"]
    pre = nn.Linear(freq, n_filter, bias=bias)
    post = nn.Linear(n_filter, freq, bias=bias)

    if init is None:
        return pre, post

    f_filter = torch.linspace(0, sr // 2, n_filter)
    delta_f = sr // 2 / n_filter
    f_freqs = torch.linspace(0, sr // 2, freq)
    down = f_filter[1:, None] - f_freqs[None, :]    # [n_filter - 1, freq]
    down = down / delta_f
    down = F.pad(down, (0, 0, 0, 1), value=1.0)     # [n_filter, freq]
    up = f_freqs[None, :] - f_filter[:-1, None]     # [n_filter - 1, freq]
    up = up / delta_f
    up = F.pad(up, (0, 0, 1, 0), value=1.0)         # [n_filter, freq]
    pre_weight = torch.max(up.new_zeros(1), torch.min(down, up))
    post_weight = pre_weight.transpose(0, 1)
    pre_weight = pre_weight / pre_weight.sum(dim=1, keepdim=True)
    post_weight = post_weight / post_weight.sum(dim=1, keepdim=True)

    if init.endswith("_fixed"):
        delattr(pre, "weight")
        delattr(post, "weight")
        pre.register_buffer("weight", pre_weight.contiguous().clone())
        post.register_buffer("weight", post_weight.contiguous().clone())
    else:
        pre.weight.data.copy_(pre_weight)
        post.weight.data.copy_(post_weight)

    return pre, post


def hz_to_erb(hz: torch.Tensor) -> torch.Tensor:
    """Converts frequency from Hertz to ERB-number scale."""
    return 21.4 * torch.log10(4.37e-3 * hz + 1)

def erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    """Converts frequency from ERB-number scale to Hertz."""
    return (10 ** (erb / 21.4) - 1) / 4.37e-3

def rf_pre_post_erb(
    freq: int,
    n_filter: int,
    init: tp.Optional[str],
    bias: bool,
    sr: int = 16_000
) -> tp.Tuple[nn.Module, nn.Module]:
    """
    Creates pre and post processing layers initialized with ERB filterbank weights.
    
    Args:
        freq: Number of input frequency bins (e.g., FFT size // 2 + 1 or just freq bins).
        n_filter: Number of filters in the filterbank.
        init: Initialization method ('linear', 'linear_fixed', or None).
        bias: Whether to include bias in Linear layers.
        sr: Sampling rate.
        
    Returns:
        A tuple of (pre_processing_layer, post_processing_layer).
    """
    assert init in [None, "erb", "erb_fixed"]
    pre = nn.Linear(freq, n_filter, bias=bias)
    post = nn.Linear(n_filter, freq, bias=bias)

    if init is None:
        return pre, post

    # 1. Create center frequencies in ERB scale
    # Convert min/max frequencies to ERB scale
    f_min_erb = hz_to_erb(torch.tensor(0.0))
    f_max_erb = hz_to_erb(torch.tensor(sr / 2))
    
    # Create linearly spaced points in ERB domain, then convert back to Hz
    erb_points = torch.linspace(f_min_erb, f_max_erb, n_filter)
    f_filter = erb_to_hz(erb_points) # Center frequencies of filters [n_filter]
    
    # 2. Calculate bandwidths (variable delta)
    # Since ERB is non-linear, delta_f is not constant.
    # We calculate the difference between adjacent center frequencies.
    # shape: [n_filter - 1]
    diffs = f_filter[1:] - f_filter[:-1] 
    
    # 3. Create frequency grid for input bins
    f_freqs = torch.linspace(0, sr // 2, freq)

    # 4. Create Triangular Filterbank Weights with variable bandwidths
    
    # 'down' slope: Defined by the current peak and the next peak.
    # Calculation: (center[i+1] - freq) / (center[i+1] - center[i])
    # Numerator shape: [n_filter - 1, freq]
    down_numerator = f_filter[1:, None] - f_freqs[None, :]
    # Denominator shape: [n_filter - 1, 1] (Broadcasting variable bandwidths)
    down = down_numerator / diffs[:, None]
    down = F.pad(down, (0, 0, 0, 1), value=1.0) # [n_filter, freq]

    # 'up' slope: Defined by the previous peak and the current peak.
    # Calculation: (freq - center[i]) / (center[i+1] - center[i]) (Note: index shifted)
    # Numerator shape: [n_filter - 1, freq]
    up_numerator = f_freqs[None, :] - f_filter[:-1, None]
    # Denominator shape: [n_filter - 1, 1]
    up = up_numerator / diffs[:, None]
    up = F.pad(up, (0, 0, 1, 0), value=1.0) # [n_filter, freq]

    # Combine slopes to form triangles
    pre_weight = torch.max(up.new_zeros(1), torch.min(down, up))
    
    # Transpose for post-processing
    post_weight = pre_weight.transpose(0, 1)

    # Normalize weights (L1 normalization to preserve energy)
    # Add epsilon to avoid division by zero if a filter covers no bins (unlikely but safe)
    pre_weight = pre_weight / (pre_weight.sum(dim=1, keepdim=True) + 1e-8)
    post_weight = post_weight / (post_weight.sum(dim=1, keepdim=True) + 1e-8)

    # 5. Assign Weights
    if init.endswith("_fixed"):
        delattr(pre, "weight")
        delattr(post, "weight")
        pre.register_buffer("weight", pre_weight.contiguous().clone())
        post.register_buffer("weight", post_weight.contiguous().clone())
    else:
        pre.weight.data.copy_(pre_weight)
        post.weight.data.copy_(post_weight)
        
    return pre, post