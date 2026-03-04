import math
import typing as tp
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from torch.nn.utils.parametrize import remove_parametrizations
from torch import Tensor

class TeacherHencer(nn.Module):
    def __init__(
        self,
        n_fft = 512,
        teacher_channe: Int = [1,64,128,64],
        channels: int = 64,
        kernel_size: tp.List[int] = [8, 3, 3],
        stride: int = 4,
        rnnformer_kwargs: tp.Dict[str, tp.Any] = dict(),
        activation: str = "ReLU",
        activation_kwargs: tp.Dict[str, tp.Any] = dict(inplace=True),
        mask: tp.Optional[str] = None,
        weight_norm: bool = False,
        normalize_final_conv: bool = False,
        pre_post_init: tp.Optional[str] = None,
        resnet: bool = False,
        domain = "none",
        **kwargs
    ):
        super().__init__()
        rnnformer_config = RNNFormerConfig(**rnnformer_kwargs)
        self.rf_ch = rnnformer_config.channels
        self.rf_freq = rnnformer_config.freq

        self.weight_norm = weight_norm
        self.resnet = resnet

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm1d

        def norm(module):
            if self.weight_norm:
                return weight_norm_fn(module)
            return module

        Act = getattr(nn, activation)

        # Encoder PreNet
        assert kernel_size[0] % stride == 0
        assert (kernel_size[0] - stride) % 2 == 0

        if domain == "none" : 
            print(f"TeacherHencer: Using default pre/post encoder/decoder.")
            self.enc_pre = DefaultPreEncoder(channels,kernel_size[0],stride,BatchNorm,Act,activation_kwargs)
            self.dec_post = DefaultPostDecoder(channels,kernel_size[0],stride,BatchNorm,Act,activation_kwargs,normalize_final_conv,mask)
        elif domain == "mag" :
            print(f"TeacherHencer: Using mag pre/post encoder/decoder.")
            self.enc_pre = MagPreEncoder(channels,kernel_size[0],stride,BatchNorm,Act,activation_kwargs)
            self.dec_post = MagPostDecoder(channels,kernel_size[0],stride,BatchNorm,Act,activation_kwargs,normalize_final_conv,mask)
        else : 
            raise RuntimeError(f"model_kwargs.domain={domain} is not supported.")

        # Encoder
        self.encoder = nn.ModuleList()
        for idx in range(1, len(kernel_size)):
            module = nn.Sequential(
                nn.Conv1d(
                    channels, channels, kernel_size[idx], 
                    padding=(kernel_size[idx] - 1) // 2, bias=False
                ),
                BatchNorm(channels),
                Act(**activation_kwargs),
            )
            self.encoder.append(module)

        # RNNFormer PreNet
        freq = n_fft // 2 // stride
        rf_pre, rf_post = rf_pre_post_lin(freq, self.rf_freq, pre_post_init, bias=False)
        self.rf_pre = nn.Sequential(
            rf_pre,
            nn.Conv1d(channels, self.rf_ch, 1, bias=False),
            BatchNorm(self.rf_ch),
        )

        # RNNFormer Blocks
        rf_list = []
        for _ in range(rnnformer_config.num_blocks):
            block = RNNFormerBlock(
                self.rf_ch, self.rf_freq,
                rnnformer_config.num_heads,
                eps=rnnformer_config.eps, weight_norm=weight_norm,
                activation=activation, activation_kwargs=activation_kwargs,
                positional_embedding=rnnformer_config.positional_embedding,
                attn_bias=rnnformer_config.attn_bias,
                post_act=rnnformer_config.post_act,
                pre_norm=rnnformer_config.pre_norm,
                p_dropout=rnnformer_config.p_dropout,
            )
            rf_list.append(block)
            rnnformer_config.positional_embedding = None
        self.rf_block = nn.ModuleList(rf_list)

        # RNNFormer PostNet
        self.rf_post = nn.Sequential(
            rf_post,
            nn.Conv1d(self.rf_ch, channels, 1, bias=False),
            BatchNorm(channels),
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for idx in range(len(kernel_size)-1, 0, -1):
            module = nn.Sequential(
                nn.Conv1d(channels*2, channels, 1, bias=False),
                BatchNorm(channels),
                Act(**activation_kwargs),
                nn.Conv1d(
                    channels, channels, kernel_size[idx],
                    padding=(kernel_size[idx] - 1) // 2, bias=False
                ),
                BatchNorm(channels),
                Act(**activation_kwargs),
            )
            self.decoder.append(module)



    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        """ 1. Remove weight_norm """
        if self.weight_norm:
            with torch.enable_grad():
                # RNNFormer
                for block in self.rf_block:
                    block.remove_weight_reparameterizations()

                # Decoder
                self.dec_post[3].remove_weight_reparameterizations()
            self.weight_norm = False

        """ 2. Merge BatchNorm into Conv
        y = (conv(x) - mean) / std * gamma + beta \
          = conv(x) * (gamma / std) + (beta - mean * gamma / std)
        <=> y = conv'(x) where
          W'[c, :, :] = W[c, :, :] * (gamma / std)
          b' = (beta - mean * gamma / std)
        """
        def merge_conv_bn(conv: nn.Module, norm: nn.Module, error_message: str = "") -> nn.Module:
            assert conv.bias is None, error_message
            std = norm.running_var.add(norm.eps).sqrt()
            conv.weight.data *= norm.weight.view(-1, 1, 1) / std.view(-1, 1, 1)
            conv.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
            return conv

        # Encoder PreNet
        conv = merge_conv_bn(self.enc_pre[0], self.enc_pre[1], "enc_pre")
        self.enc_pre = nn.Sequential(conv, self.enc_pre[2])

        # Encoder
        new_encoder = nn.ModuleList()
        for idx, module in enumerate(self.encoder):
            conv = merge_conv_bn(module[0], module[1], f"enc.{idx}")
            new_module = nn.Sequential(
                conv,       # Conv-BN Merged
                module[2],  # Activation
            )
            new_encoder.append(new_module)
        self.encoder = new_encoder

        # RNNFormer PreNet
        conv = merge_conv_bn(self.rf_pre[1], self.rf_pre[2], "rf_pre")
        self.rf_pre = nn.Sequential(
            self.rf_pre[0],     # Linear
            conv,               # Conv-BN Merged
        )

        # RNNFormer PostNet
        conv = merge_conv_bn(self.rf_post[1], self.rf_post[2], "rf_post")
        self.rf_post = nn.Sequential(
            self.rf_post[0],    # Linear
            conv,               # Conv-BN Merged
        )

        # Decoder
        new_decoder = nn.ModuleList()
        for idx, module in enumerate(self.decoder):
            conv1 = merge_conv_bn(module[0], module[1], f"dec.{idx}.0")
            conv2 = merge_conv_bn(module[3], module[4], f"dec.{idx}.1")
            new_module = nn.Sequential(
                conv1,      # Conv-BN Merged
                module[2],  # Activation
                conv2,      # Conv-BN Merged
                module[5],  # Activation
            )
            new_decoder.append(new_module)
        self.decoder = new_decoder

        # Decoder PostNet
        conv = merge_conv_bn(self.dec_post[0], self.dec_post[1], "dec_post")
        self.dec_post = nn.Sequential(
            conv,               # Conv-BN Merged
            self.dec_post[2],   # Activation
            self.dec_post[3]    # Transposed Convolution
        )

    def flatten_parameters(self):
        for rf in self.rf_block:
            rf.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_list = []
        for block in self.rf_block:
            cache_list.append(block.initialize_cache(x))
        return cache_list

    def forward(self, spec_noisy: Tensor, hs = []) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # spec_noisy: [B, F, T, 2]
        cache_in_list = hs 
        cache_out_list = []
        if len(cache_in_list) == 0:
            cache_in_list = [None for _ in range(len(self.rf_block))]

        B, FREQ, T, _ = spec_noisy.shape
        x = spec_noisy.permute(0, 2, 3, 1)       # [B, T, 2, F]
        x = x.reshape(B*T, 2, FREQ)     # [B*T, 2, F]

        # Encoder PreNet
        x = self.enc_pre(x)
        encoder_outs = [x]

        # Encoder
        for module in self.encoder:
            x_in = x
            x = module(x)
            encoder_outs.append(x)      # [B*T, C, F']
            if self.resnet:
                x = x.add_(x_in)

        # RNNFormer
        x_in = x
        x = self.rf_pre(x)              # [B*T, C, F']
        _, C, _FREQ = x.shape
        x = x.view(B, T, C, _FREQ)      # [B, T, C, F']
        x = x.permute(1, 0, 3, 2)       # [T, B, F', C]
        x = x.contiguous()
        for block, cache_in in zip(self.rf_block, cache_in_list):
            x, cache_out = block(x, cache_in)   # [T, B, F', C]
            cache_out_list.append(cache_out)
        x = x.permute(1, 0, 3, 2)       # [B, T, C, F']
        x = x.reshape(B*T, C, _FREQ)    # [B*T, C, F']
        x = self.rf_post(x)             # [B*T, C, F']
        if self.resnet:
            x = x.add_(x_in)

        import pdb; pdb.set_trace()

        # Decoder
        for module in self.decoder:
            x_in = x
            x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
            x = module(x)                                       # [B*T, C, F] or [B*T, 2, F]
            if self.resnet:
                x = x.add_(x_in)

        # Decoder PostNet
        x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
        spec_hat = self.dec_post(spec_noisy,x)                           # [B*T, 2, F]

        return spec_hat, cache_out_list
    
    # TODO
    def to_onnx(self, output_fp, device=torch.device("cpu")):
        print("ONNX export is not implemented yet for TeacherHencer.")
        """
        torch.onnx.export(
            self,
            (dummy_input, dummy_states, dummy_pe_state),
            output_fp,
            verbose=False,
            opset_version=16,
            input_names=["inputs", "gru_state_in", "pe_state_in"],
            output_names=["outputs", "gru_state_out", "pe_state_out"],
            dynamo=False
        )
        """

class TeacherHencerWrapper(nn.Module):
    def __init__(self,
            n_fft: int = 512,
            hop_size : int = 256,
            input_compression: float = 0.3,
            eps = 1e-7,
            **kwargs):
        super().__init__()

        self.model = TeacherHencer(n_fft=n_fft,**kwargs)

        # Compressed STFT and discard_last_freq_bin
        self.frame_size = n_fft
        self.hop_size = hop_size
        self.compression = input_compression
        self.eps = eps
        
        self.window = torch.zeros(self.frame_size)
        if self.frame_size // self.hop_size == 4 : 
            n = torch.arange(self.frame_size)
            self.window = 0.5 * (1.0 - torch.cos(2.0 * np.pi * n / self.frame_size))
            
            energy_sum = torch.sum(self.window ** 2) / self.hop_size
            self.window /= torch.sqrt(energy_sum)
        elif self.frame_size // self.hop_size == 2 :
            n = torch.arange(self.frame_size)
            self.window = torch.sin(np.pi * (n + 0.5) / self.frame_size)
            
            energy_sum = torch.sum(self.window ** 2) / self.hop_size
            self.window /= torch.sqrt(energy_sum)
        else :
            raise RuntimeError(f"Not supported frame_size // hop_size {self.frame_size}//{self.hop_size}")
        
    def _to_spec(self,x):
        B,L = x.shape
        # X : [B,F,T,2]
        X = torch.stft(x, n_fft = self.frame_size, hop_length = self.hop_size, window=self.window.to(x.device),return_complex=False)

        # Discard last freq bin
        # X : [B,F-1,T,2]
        X = X[:, :-1, :, :]

        # Magnitudfe Compression
        mag = torch.linalg.norm(X, dim=-1, keepdim=True).clamp(min=self.eps)
        X = X * mag.pow(self.compression - 1.0)
        return X

    def _to_signal(self, Y):
        Y = Y[...,0] + 1j * Y[...,1]  

        # Magnitude Decompression
        mag_compressed = Y.abs()
        Y = Y* mag_compressed.pow((1.0 / self.compression) - 1.0)

        # Append last freq bin of zeros
        Y = F.pad(Y, (0,0,0,1), "constant", 0.0)  # [B,F,T]

        y = torch.istft(Y, self.frame_size, self.hop_size, self.frame_size, self.window.to(Y.device))
        
        return y

    def forward(self, x):
        B,L = x.shape
        
        X = self._to_spec(x)
        Y,_ = self.model(X)

        y = self._to_signal(Y)
        return y[:, :L]
    
    def to_onnx(self, output_fp, device=torch.device("cpu")):
        self.model.to_onnx(output_fp, device)


class StridedConv1d(nn.Conv1d):
    """Same as nn.Conv1d with stride > 1.
    We just want to show that MAC of StridedConv is not Cin x Cout x K x T (ptflops calculation),
    but Cin x Cout x K x (T / S) where K: kernel_size, T: time, S: stride."""
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1,
        groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        device=None, dtype=None
    ):
        assert kernel_size % stride == 0, (
            f'kernel_size k and stride s must satisfy k=(2n+1)s, but '
            f'got k={kernel_size}, s={stride}. Use naive Conv1d instead.'
        )
        assert groups == 1, (
            f'groups must be 1, but got {groups}. '
            f'Use naive Conv1d instead.'
        )
        assert dilation == 1, (
            f'dilation must be 1, but got {dilation}. '
            f'Use naive Conv1d instead.'
        )
        assert padding_mode == 'zeros', (
            f'Only `zeros` padding mode is supported for '
            f'StridedConv1d, but got {padding_mode}. '
            f'Use naive Conv1d instead.'
        )
        self.original_stride = stride
        self.original_padding = padding
        super().__init__(
            in_channels*stride, out_channels, kernel_size//stride,
            stride=1, padding=0, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
            device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, Ci, Ti] -> conv1d -> [B, Co, Ti // s]
        <=> x: [B, Ci, Ti] -> reshape to [B, Ci*s, Ti//s] -> conv1d -> [B, Co, Ti//S]"""
        stride = self.original_stride
        padding = self.original_padding
        x = F.pad(x, (padding, padding))
        B, C, T = x.shape
        x = x.view(B, C, T//stride, stride).permute(0, 3, 1, 2).reshape(B, C*stride, T//stride)
        return super().forward(x)


class ScaledConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        *args,
        normalize: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.scale = nn.Parameter(torch.ones(1))
        self.weight_norm = True

    def remove_weight_reparameterizations(self):
        if self.normalize:
            weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
        else:
            weight = self.weight * self.scale
        self.weight.data.copy_(weight)
        self.weight_norm = False
        self.scale = None

    def forward(self, x: Tensor) -> Tensor:
        if self.weight_norm:
            if self.normalize:
                weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
            else:
                weight = self.weight * self.scale
        else:
            weight = self.weight
        return F.conv_transpose1d(
            x, weight, self.bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, dilation=self.dilation,
        )


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
    
class DefaultPreEncoder(nn.Module):
    def __init__(self, channels,kernel_size,stride,BatchNorm,Act,activation_kwargs):
        super().__init__()
        self.enc_pre = nn.Sequential(
            StridedConv1d(  # in_channels = 2 = [real, imag]
                2, channels, kernel_size, stride=stride,
                padding=(kernel_size - stride) // 2, bias=False
            ),
            BatchNorm(channels),
            Act(**activation_kwargs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_pre(x)
    
class DefaultPostDecoder(nn.Module):
    def __init__(self, channels,kernel_size,stride,BatchNorm,Act,activation_kwargs,normalize_final_conv,mask):
        super().__init__()

        # Decoder PostNet
        # out_channels = 2 = [real, imag] of the mask
        upsample = ScaledConvTranspose1d(
            channels, 2, kernel_size, stride=stride,
            bias=True,
            padding=(kernel_size - stride) // 2,
            normalize=normalize_final_conv,
        )
        self.dec_post = nn.Sequential(
            nn.Conv1d(channels*2, channels, 1, bias=False),
            BatchNorm(channels),
            Act(**activation_kwargs),
            upsample
        )
        if mask is None:
            self.mask = nn.Identity()
        elif mask == "sigmoid":
            self.mask = nn.Sigmoid()
        elif mask == "tanh":
            self.mask = nn.Tanh()
        else:
            raise RuntimeError(f"model_kwargs.mask={mask} is not supported.")
        
    def forward(self, spec_noisy:Tensor, z: Tensor) -> Tensor:
        z = self.dec_post(z)

        B,T,_,FREQ = spec_noisy.shape

        z = z.reshape(B, T, 2, FREQ).permute(0, 3, 1, 2)    # [B, F, T, 2]

        # Mask
        mask = self.mask(z).contiguous()

        spec_hat = torch.stack(
            [
                spec_noisy[..., 0] * mask[..., 0] - spec_noisy[..., 1] * mask[..., 1],
                spec_noisy[..., 0] * mask[..., 1] + spec_noisy[..., 1] * mask[..., 0],
            ],
            dim=3
        )   # [B, F, T, 2]
        return spec_hat
    
class MagPreEncoder(nn.Module):
    def __init__(self, channels,kernel_size,stride,BatchNorm,Act,activation_kwargs):
        super().__init__()
        self.enc_pre = nn.Sequential(
            StridedConv1d(  # in_channels = 2 = [real, imag]
                1, channels, kernel_size, stride=stride,
                padding=(kernel_size - stride) // 2, bias=False
            ),
            BatchNorm(channels),
            Act(**activation_kwargs),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x :  [B*T, 2, F] -> mag : [B*T, 1, F]
        mag = torch.linalg.norm(x, dim=1, keepdim=True)
        return self.enc_pre(mag)
    
class MagPostDecoder(nn.Module):
    def __init__(self, channels,kernel_size,stride,BatchNorm,Act,activation_kwargs,normalize_final_conv,mask):
        super().__init__()
        upsample = ScaledConvTranspose1d(
            channels, 1, kernel_size, stride=stride,
            bias=True,
            padding=(kernel_size - stride) // 2,
            normalize=normalize_final_conv,
        )
        self.dec_post = nn.Sequential(
            nn.Conv1d(channels*2, channels, 1, bias=False),
            BatchNorm(channels),
            Act(**activation_kwargs),
            upsample
        )
        if mask is None:
            self.mask = nn.Identity()
        elif mask == "sigmoid":
            self.mask = nn.Sigmoid()
        elif mask == "tanh":
            self.mask = nn.Tanh()
        else:
            raise RuntimeError(f"model_kwargs.mask={mask} is not supported.")
        
    def forward(self, spec_noisy:Tensor, z: Tensor) -> Tensor:
        # spec_noisy : [B,F,T,2]

        z = self.dec_post(z)
        B,FREQ,T,_ = spec_noisy.shape
        z = z.reshape(B, T, 1, FREQ).permute(0, 3, 1, 2)    # [B, F, T, 1]
        # Mask
        mask = self.mask(z).contiguous()

        mag_noisy = torch.linalg.norm(spec_noisy, dim=-1)  # [B, F, T]
        noisy_phase = torch.atan2(spec_noisy[..., 1], spec_noisy[..., 0])  # [B, F, T]
        mag_hat = mag_noisy * mask[..., 0]  # [B, F, T]
        spec_re = mag_hat * torch.cos(noisy_phase)
        spec_im = mag_hat * torch.sin(noisy_phase)
        spec_hat = torch.stack([spec_re, spec_im], dim=-1)  # [B, F, T, 2]

        return spec_hat 



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

def rf_pre_post_equ(
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




def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/fastenhancer/t.yaml")
    model = Model(**hparams["model_kwargs"])
    wav_out, spec_out = model(x)
    (wav_out.sum() + spec_out.sum()).backward()
    print(spec_out.shape)

    model.remove_weight_reparameterizations()
    model.flatten_parameters()
    total_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"Number of total parameters: {total_params}")
    # for n, p in model.named_parameters():
    #     print(n, p.shape)


if __name__ == "__main__":
    test()