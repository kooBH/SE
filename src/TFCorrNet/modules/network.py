import torch as th
import torch.nn as nn
import math
import numpy
from utils.decorators import *

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int d_model: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, d_model, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert d_model % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, pos_k, mask=None):
        """Compute 'Scaled Dot Product Attention'.

        :param th.Tensor mask: (batch, time1, time2)
        :param th.nn.Dropout dropout:
        :return th.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)   #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = th.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = th.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(th.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = th.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = th.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = th.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))  # (batch, time1, d_model)
    


class GCFN(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, Layer_scale_init: float=1.0e-5):
        super().__init__()
        expan_factor = 6
        # self.depthwise = th.nn.Conv1d(d_model*expan_factor, d_model*expan_factor, 3, padding=1, groups=d_model*expan_factor)
        self.net = th.nn.Sequential(
            th.nn.LayerNorm(d_model),
            th.nn.Linear(d_model, d_model*expan_factor),
            th.nn.GLU(),
            th.nn.Dropout(dropout_rate),
            th.nn.Linear(d_model*expan_factor//2, d_model),
            th.nn.Dropout(dropout_rate))
        self.gate = th.nn.Sequential(
            th.nn.Linear(d_model, d_model), 
            th.nn.Sigmoid())
        # self.Layer_scale = LayerScale(dims=3, input_size=d_model, Layer_scale_init=Layer_scale_init)
        
    def forward(self, x):
        down_len = x.shape[1]//4
        x = x.permute([0, 2, 1])
        x_down = th.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute([0, 2, 1])
        x_down = x_down.permute([0, 2, 1])

        x_down = self.net(x_down)

        x_down = x_down.permute([0, 2, 1])
        x_downup = th.nn.functional.upsample(input=x_down, size=x.shape[1])
        x_downup = x_downup.permute([0, 2, 1])
        
        x = x + self.gate(x)*x_downup

        return x
    
    
class EGA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout_rate: float):
        super().__init__()
        self.block = th.nn.ModuleDict({
            'self_attn': MultiHeadAttention(
                n_head=n_head, d_model=d_model, dropout_rate=dropout_rate),
            'linear': th.nn.Sequential(
                th.nn.Linear(in_features=d_model, out_features=d_model), 
                th.nn.Sigmoid())
        })
    
    def forward(self, x: th.Tensor, pos_k: th.Tensor):
        """
        Compute encoded features.
            :param th.Tensor x: encoded source features (batch, max_time_in, size)
            :param th.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[th.Tensor, th.Tensor]
        """
        x = x.permute([0, 2, 1])
        down_len = pos_k.shape[0]
        x_down = th.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute([0, 2, 1])
        x_down = x_down.permute([0, 2, 1])
        x_down = self.block['self_attn'](x_down, pos_k, None)
        x_down = x_down.permute([0, 2, 1])
        x_downup = th.nn.functional.upsample(input=x_down, size=x.shape[1])
        x_downup = x_downup.permute([0, 2, 1])
        x = x + self.block['linear'](x) * x_downup

        return x



class CLA(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim*2)
        self.GLU = nn.GLU(dim=-1)
        self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, padding='same', groups=input_dim)
        self.linear2 = nn.Sequential(
            # th.nn.GELU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.Dropout(dropout_rate))
        # self.Layer_scale = LayerScale(dims=3, input_size=input_dim, Layer_scale_init=Layer_scale_init)
    
    def forward(self, x):
        y = self.layer_norm(x)
        # y = self.linear1(y)
        y = self.linear1(y)
        y = self.GLU(y)
        y = y.permute([0, 2, 1]) # B, F, T
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1) # B, T, 2F        
        y = self.linear2(y)
        
        return x + y


class GlobalBlock(th.nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout_rate: float):
        super().__init__()
        self.block = th.nn.ModuleDict({
            'ega': EGA(
                d_model=d_model, n_head=n_head, dropout_rate=dropout_rate),
            'gcfn': GCFN(d_model=d_model, dropout_rate=dropout_rate)
        })
    
    def forward(self, x: th.Tensor, pos_k: th.Tensor):
        """
        Compute encoded features.
            :param th.Tensor x: encoded source features (batch, max_time_in, size)
            :param th.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[th.Tensor, th.Tensor]
        """
        x = self.block['ega'](x, pos_k)
        x = self.block['gcfn'](x)

        return x


class LocalBlock(th.nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.block = th.nn.ModuleDict({
            'cla': CLA(d_model, kernel_size, dropout_rate),
            'gcfn': GCFN(d_model, dropout_rate)
        })
    
    def forward(self, x: th.Tensor):
        x = self.block['cla'](x)
        x = self.block['gcfn'](x)

        return x

class GLformer(nn.Module):
    
    def __init__(self, d_model: int, n_head: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.global_block = GlobalBlock(d_model, n_head, dropout_rate)
        self.local_block = LocalBlock(d_model, kernel_size, dropout_rate)

    def forward(self, x: th.tensor, pos_k: th.tensor):
        #* B, L, C
        x = self.global_block(x, pos_k)
        x = self.local_block(x)        
        
        return x