import torch
from torch import Tensor, nn
import torch.nn.functional as nF
import numpy as np
from functools import partial
import math

### Encoder
class DFNConvLayer(nn.Sequential):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1,1),
                 padding=[0,0],
                 bias=False,
                 transpose=False,
                 output_padding=0,
                 groups = -1
                 ):

        # Depthwise seperable convolution
        separable = True
        if groups == -1:
            groups = math.gcd(in_channels, out_channels)

        # Unable to use separable convolution
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False

        layers = []

        if not transpose:
            layers.append(nn.Conv2d(in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups)
            )
        else :
            layers.append(nn.ConvTranspose2d(in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias,
                    groups=groups)
            )

        if separable:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1,bias=False))
        
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())

        super().__init__(*layers)

DFNTrConvLayer = partial(DFNConvLayer, transpose=True)


# Conv With LayerNorm
class ConvV1(nn.Sequential):
    def __init__(self, 
                in_channels,
                out_channels,
                hidden_size,
                kernel_size,
                stride=(1,1),
                padding=[0,0],
                bias=False,
                transpose=False,
                output_padding=0,
                groups = -1
                ):

        # Depthwise seperable convolution
        separable = True
        if groups == -1:
            groups = math.gcd(in_channels, out_channels)

        # Unable to use separable convolution
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False

        layers = []

        if not transpose:
            layers.append(nn.Conv2d(in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups)
            )
        else :
            layers.append(nn.ConvTranspose2d(in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias,
                    groups=groups)
            )

        layers.append(nn.LayerNorm(hidden_size))
        if separable:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1,bias=False))
        
        layers.append(nn.PReLU())

        super().__init__(*layers)


## Linear

class GroupedLinear(nn.Module):
    input_size: int
    hidden_size: int
    groups: int
    def __init__(self, 
        input_size: int, 
        hidden_size: int,
        groups: int = 1
        ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        # einsum operation ->  iterate bmm over groups
        self.bmm = nn.Linear(self.input_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i in range(self.groups):
            outputs.append(self.bmm(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)

        return output

### RNN

class SqueezedGRU(nn.Module): 
    def __init__(self,input_size,hidden_size,output_size = None, groups=1, num_layers=1) :
        super(SqueezedGRU, self).__init__()

        self.emb_in = nn.Sequential(GroupedLinear(input_size, hidden_size, groups),nn.ReLU())

        self.gru = nn.GRU(hidden_size,hidden_size,num_layers = num_layers, batch_first=True)

        if output_size is not None :
            self.emb_out = nn.Sequential(GroupedLinear(hidden_size, output_size, groups),nn.ReLU())
        else :
            self.emb_out = nn.Identity()

        # disabled emb_out
        # disabled gru skip

        self.activation = nn.ReLU()

    def forward(self,x,h = None) :
        # x : [B,T,F]
        x = self.emb_in(x)
        x,h = self.gru(x,h)
        x = self.emb_out(x)
        return x,h

### Emb ###
class EmbCat(nn.Module):
    def __init__(self):
        super(EmbCat, self).__init__()

    def forward(self, *inputs):
        # a : [...,Fa]
        # b : [...,Fb]
        # return : [...,Fa+Fb] ]
        return torch.cat(inputs,dim=-1)

class EmbAdd(nn.Module) :
    def __init__(self):
        super(EmbAdd, self).__init__()
    
    def forward(self, a,b):
        # a : [...,F]
        # b : [...,F]
        # return : [...,F]
        return a+b

class EmbFC(nn.Module) : 
    def __init__(self,hidden_size):
        super(EmbFC,self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size,bias=False)
        self.activation = nn.ReLU()

    def forward(self,*inputs):
        # a : (B,T,Fa)
        # b : (B,T,Fb)
        # return : (B,T,F)
        emb = torch.cat(inputs,dim=-1)

        emb = self.fc(emb)
        emb = self.activation(emb)

        return emb

"""
current : v4
v1 : default
v2 : added residual connection and reLU activation after SA
v3 : FSA for each feature
v4 : linear layer for each feature
v5 : share FSA
"""
class EmbFSA(nn.Module) :
    def __init__(self, erb_size, df_size, erb_f, df_f,hidden_size, type_PE = "PositionalEncoding",num_heads=4,type_FSA_activation = "SiLU"):
        super(EmbFSA, self).__init__()

        self.FSA = FSA(erb_size,num_heads=num_heads,type_PE=type_PE,type_activation=type_FSA_activation)
        #self.FSA_df = FSA(df_size, num_heads=num_heads,type_PE=type_PE)
        self.linear_erb = nn.Linear(erb_f*erb_size, hidden_size, bias=False)
        self.linear_df = nn.Linear(df_f*df_size, hidden_size, bias=False)
        self.activation = nn.ReLU()

    def forward(self, a,b):
        # a : (B,C,T,Fa)
        # b : (B,C,T,Fb)
        # return : (B,T,F)

        # B,C,T,Fa -> B*T,Fa,C
        B,C,T,Fa = a.shape
        a = a.permute(0,2,3,1)
        a = a.reshape(B*T,Fa,C)
        a = self.FSA(a)
        # B*T,Fa,C -> B,T,Fa*C
        a = a.reshape(B,T,-1)

        # B,C,T,Fb -> B*T,Fb,C
        B,C,T,Fb = b.shape
        b = b.permute(0,2,3,1)
        b = b.reshape(B*T,Fb,C)
        b = self.FSA(b)
        # B*T,Fb,C -> B,T,Fb*C
        b = b.reshape(B,T,-1)

        # B,T,(Fa + Fb) -> B,T,F
        a = self.linear_erb(a)
        b = self.linear_df(b)
        #print(f"a.shape : {a.shape}, b.shape : {b.shape}")
        emb = torch.cat((a,b),dim=-1)
        #print(f"emb.shape : {emb.shape}")
        emb = self.activation(emb)
        return emb
    
class FSA(nn.Module) :
    def __init__(self, hidden_size, type_PE = "PositionalEncoding",num_heads=4, type_activation = "SiLU",**kwargs):
        super(FSA, self).__init__()

        if type_PE == "PositionalEncoding"  :
            self.PE = PositionalEncoding(hidden_size)
        elif type_PE == "Identity" :
            self.PE = nn.Identity()

        self.SA = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.SA_norm = nn.LayerNorm(hidden_size)
        self.linear_norm = nn.LayerNorm(hidden_size)

        if type_activation == "SiLU" : 
            self.activation_norm = nn.SiLU()
        elif type_activation == "ReLU" :
            self.activation_norm = nn.ReLU()

    def forward(self, x):
        # x : (B,T,D)

        x = self.PE(x)
        emb = self.SA_norm(x)
        emb = self.SA(emb, emb, emb)[0]
        emb = x + emb

        emb2 = self.linear_norm(emb)
        emb2 = self.linear(emb2)
        emb2 = self.activation_norm(emb2)
        emb = emb + emb2

        return emb

### Postional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # 짝수
        pe[0, :, 1::2] = torch.cos(position * div_term)  # 홀수

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : (B, T, D)
        x = x + self.pe[:, :x.size(1), :] 
        return self.dropout(x)

### Reshape
class Flatter(nn.Module) :
    def __init__(self):
        super(Flatter, self).__init__()

    def forward(self, x):
        # x : [B,T,C*F//2]
        x = x.permute(0,2,3,1)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        return x
    
### Inter Mask 
class Identity2arg(nn.Module):
    def __init__(self):
        super(Identity2arg, self).__init__()

    def forward(self, a, b):
        return a,b

class Identity3arg(nn.Module):
    def __init__(self):
        super(Identity3arg, self).__init__()

    def forward(self, a, b, c):
        return a,b,c
    
class InterMaskLinear(nn.Module):
    def __init__(self, erb_size, df_size,type_activation = "None",bias = False):
        super(InterMaskLinear, self).__init__()

        input_size = erb_size + df_size
        self.fc_erb = nn.Linear(input_size, erb_size,bias=bias)
        self.fc_df = nn.Linear(input_size, df_size,bias=bias)
        
        if type_activation == "ReLU":
            self.activation = nn.ReLU()
        else : 
            self.activation = nn.Identity()

        self.activation_erb = nn.Sigmoid()
        self.activation_df = nn.Sigmoid()

    def forward(self, m_erb, m_df):
        # a : [B,1,T,Ferb]
        # b : [B,T,Fdf,N]
        # return : [B,T,Ferb], [B,T,Fdf,N]
        B,T,F,N = m_df.shape
        m_df = m_df.reshape(B,T,-1)  # [B,T,Fdf*N]
        m_erb = m_erb.reshape(B,T,-1)
        emb = torch.cat((m_erb,m_df),dim=-1)
        emb = self.activation(emb)

        m_erb = self.fc_erb(emb)
        m_erb = self.activation_erb(m_erb)
        m_erb = m_erb.reshape(B,1,T,-1)

        m_df = self.fc_df(emb)
        m_df = self.activation_df(m_df)
        m_df = m_df.reshape(B,T,F,N)

        return m_erb, m_df

