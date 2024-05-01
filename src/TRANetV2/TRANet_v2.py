import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
        bias=True
    ):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups,bias=bias)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups,bias=bias)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

        if bias : 
            nn.init.constant_(self.real_conv.bias, 0.)
            nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)
            
            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        return out


def complex_cat(inp, dim=1):
    reals, imags = torch.chunk(inp, 2, dim)
    return reals, imags

class ComplexLinearProjection(nn.Module):
    def __init__(self):
        super(ComplexLinearProjection, self).__init__()

    def forward(self, real, imag):
        """
        real, imag: B C F T
        """
        #inputs = torch.cat([real, imag], 1)
        #outputs = self.clp(inputs)
        #real, imag = outputs.chunk(2, dim=1)
        outputs = torch.sqrt(real**2+imag**2+1e-8)
        return outputs

class PhaseEncoderV1(nn.Module):
    def __init__(self, in_channels=1, out_channels = 4,type_norm="None", alpha=0.5):
        super(PhaseEncoderV1, self).__init__()
        self.complexnn = nn.Sequential(
                    nn.ConstantPad2d((0, 0, 0, 0), 0.0),
                    ComplexConv2d(in_channels, out_channels, (1, 3))
                )
        self.clp = ComplexLinearProjection()
        self.alpha = alpha

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.complexnn(X)
        real, imag = complex_cat(outs, dim=1)
        amp = self.clp(real, imag)
        return amp**self.alpha

class PhaseEncoderV2(nn.Module):
    def __init__(self, in_channels = 1, out_channels=4, type_norm = "BatchNorm2d"):
        super(PhaseEncoderV2, self).__init__()
        self.complexnn = ComplexConv2d(in_channels, out_channels, (1, 3))

        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels*2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels*2,track_running_stats=True)
        self.cplx2real = nn.Conv2d(out_channels*2,out_channels,1)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.complexnn(X)
        outs = self.norm(outs)
        outs = self.cplx2real(outs)
        return outs
    
class PhaseEncoderV3(nn.Module):
    def __init__(self, out_channels, in_channels=1,type_norm = "BatchNorm2d",bias=True):
        super(PhaseEncoderV3, self).__init__()
        self.complexnn_depth = ComplexConv2d(in_channels, in_channels, (1, 3))
        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(in_channels*2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(in_channels*2,track_running_stats=True)
        self.complexnn_point = ComplexConv2d(in_channels, out_channels, (1, 1),bias=bias)
        self.cplx2real = nn.Conv2d(out_channels*2,out_channels,1,bias=bias)

    def forward(self, X):
        """
        X : [B,M(re,im),F,T]
        """
        outs = self.complexnn_depth(X)
        outs = self.norm(outs)
        outs = self.complexnn_point(outs)
        outs = self.cplx2real(outs)
        return outs
    
class PhaseEncoderV4(nn.Module):
    def __init__(self, out_channels, in_channels=2,type_norm = "BatchNorm2d"):
        super(PhaseEncoderV4, self).__init__()
        self.conv1= nn.Conv2d(in_channels, out_channels*2, (1, 3))

        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels*2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels*2,track_running_stats=True)
        self.conv2= nn.Conv2d(out_channels*2, out_channels, (1, 3),padding=(0,1))

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.conv1(X)
        outs = self.norm(outs)
        outs = self.conv2(outs)
        return outs
    
class PhaseEncoderV5(nn.Module):
    def __init__(self, out_channels, in_channels=1,type_norm = "BatchNorm2d",bias=True):
        super(PhaseEncoderV5, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, 4*in_channels, (1, 3))
        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(4*in_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(in_channels,track_running_stats=True)
        self.pw_conv = nn.Conv2d(4*in_channels, out_channels, (1, 1),bias=bias)

    def forward(self, X):
        """
        X : [B,M(re,im),F,T]
        """
        outs = self.dw_conv(X)
        outs = self.norm(outs)
        outs = self.pw_conv(outs)
        return outs

def is_tracing():
    # Taken for pytorch for compat in 1.6.0
    """
    Returns ``True`` in tracing (if a function is called during the tracing of
    code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return torch._C._is_tracing()

#define custom_atan2 to support onnx conversion
def custom_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / (x + 1e-6))
    ans += ((y > 0) & (x < 0)) * pi
    ans -= ((y < 0) & (x < 0)) * pi
    ans *= 1 - ((y > 0) & (x == 0)) * 1.0
    ans += ((y > 0) & (x == 0)) * (pi / 2)
    ans *= 1 - ((y < 0) & (x == 0)) * 1.0
    ans += ((y < 0) & (x == 0)) * (-pi / 2)
    return ans

class FirstDownConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,type_norm=None,type_activation="ReLU"):
        super(FirstDownConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif type_activation == "ELU":
            self.activation = torch.nn.ELU()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups,type_norm = "BatchNorm2d"):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()

        self.conv_depth = nn.Conv2d(
            in_channels=out_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.bn_depth = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_point(x)
        x = self.norm_point(x)
        x = F.relu(x)
        x = self.conv_depth(x)
        x = self.bn_depth(x)
        x = F.relu(x)
        return x



class TS_NeXt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups,type_norm = "BatchNorm2d",type_activation="GELU"):
        super(TS_NeXt, self).__init__()

        self.conv_depth_1 = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=in_channels,
            kernel_size=7,
            padding=3,
            groups=groups,
        )

        self.conv_depth_2 = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.conv_point_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #self.conv_point_2 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)

        self.LN = nn.GroupNorm(1, in_channels, eps=1e-8)

        if type_norm == "BatchNorm2d" : 
            self.norm_depth= nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_depth= nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_depth= nn.Identity()

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "GELU" : 
            self.activation = torch.nn.GELU()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()

    def forward(self, x):
        x_in = x
        x = self.conv_depth_1(x)
        x = self.LN(x)
        x = self.conv_point_1(x)
        x = self.activation(x)
        x = x_in + x
        #x = self.conv_point_2(x)
        #x = F.gelu(x)
        x = self.conv_depth_2(x)
        x = self.norm_depth(x)
        x = self.activation(x)
        return x
    


class DepthwiseNext(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups,type_norm = "BatchNorm2d",type_activation="ReLU"):
        super(DepthwiseNext, self).__init__()
        self.conv_depth = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.conv_point_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #self.conv_point_2 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)

        if type_norm == "BatchNorm2d" : 
            self.norm_depth= nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_depth= nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_depth= nn.Identity()

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif type_activation == "ELU":
            self.activation = torch.nn.ELU()
        elif type_activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.norm_depth(x)
        x = self.conv_point_1(x)
        #x = self.conv_point_2(x)
        #x = F.gelu(x)
        x = self.activation(x)
        return x
    
# Gate Convolution Layer
class GConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,  groups=1,type_norm = "BatchNorm2d",type_activation="ReLU"):
        super(GConv,self).__init__()
        
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  groups=groups)
        self.m = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  groups=groups),
            nn.Sigmoid()
            )
        
        if type_norm == "BatchNorm2d" : 
            self.norm= nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm= nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm= nn.Identity()

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif type_activation == "ELU":
            self.activation = torch.nn.ELU()
        elif type_activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()
        
    def forward(self, x): 
        m = self.m(x)
        x = self.c(x)

        y = m*x
        y = self.norm(y)
        y = self.activation(y)
        
        return y
        


class FGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(FGRUBlock, self).__init__()
        self.GRU = nn.GRU(
            in_channels, hidden_size, batch_first=True, bidirectional=True
        )
        # the GRU is bidirectional -> multiply hidden_size by 2
        self.conv = nn.Conv2d(hidden_size * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x):
        """x has shape (batch * timesteps, number of features, feature_size)"""
        # We want the FGRU to consider the F axis as the one that unrolls the sequence,
        #  C the input_size, and BT the effective batch size --> x_.shape == (B,C,T,F)
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        y, h = self.GRU(x_)  # x_.shape == (BT,F,C)
        y = y.reshape(B, T, F, self.hidden_size * 2)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.conv(output)
        output = self.bn(output)
        return self.relu(output)

class TGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, **kwargs):
        super(TGRUBlock, self).__init__()

        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        # We want the GRU to consider the T axis as the one that unrolls the sequence,
        #  C the input_size, and BS the effective batch size --> x_.shape == (BS,T,C)
        # B = batch_size, T = time_steps, C = num_channels, S = feature_size
        #  (using S for feature_size because F is taken by nn.functional)
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x1 = x.permute(0, 3, 2, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 3, 2, 1)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output, rnn_state
    
class TLSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, **kwargs):
        super(TLSTMBlock, self).__init__()

        self.GRU= nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        # We want the GRU to consider the T axis as the one that unrolls the sequence,
        #  C the input_size, and BS the effective batch size --> x_.shape == (BS,T,C)
        # B = batch_size, T = time_steps, C = num_channels, S = feature_size
        #  (using S for feature_size because F is taken by nn.functional)
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x1 = x.permute(0, 3, 2, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 3, 2, 1)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output, rnn_state
    
class T_FGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels,**kwargs):
        super(T_FGRUBlock, self).__init__()
        self.hidden_size = hidden_size

        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    # TODO : if it works, implement forward with hidden like TGRUBlock
    def forward(self, x, rnn_state=None):
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x_ = x.reshape(B * C, T, F)  # x_.shape == (BC,T,F)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y = y_.reshape(B, C, T, F)  # y1.shape == (B,F,T,C)

        output = self.bn(y)
        output = self.relu(output)
        # for now
        #return output, rnn_state
        return output

class FSABlock(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(FSABlock,self).__init__()
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        y,h = self.SA(x_,x_,x_)
        
        y = y.reshape(B, T, F, C)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.bn(output)
        output = self.relu(output)
        return output

#  Channel wise Attetnion
class CSABlock(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(CSABlock,self).__init__()
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,C,F]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 1, 3)  # x_.shape == (B,T,C,F)
        x_ = x_.reshape(B * T, C, F)
        
        y,h = self.SA(x_,x_,x_)
        
        y = y.reshape(B, T, C, F)
        output = y.permute(0, 2, 1, 3)  # output.shape == (B,C,T,F)
        output = self.bn(output)
        output = self.relu(output)
        return output   

# Frequence Axis Transformer
class FATBlock(nn.Module):
    def __init__(self,in_channels,dropout=0.0) :
        super(FATBlock,self).__init__()
        
        self.T = nn.Transformer(in_channels, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        y = self.T(x_,x_)
        
        y = y.reshape(B, T, F, C)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
class TrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TrCNN, self).__init__()
        self.out_channels = out_channels
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_point = nn.BatchNorm2d(out_channels)
        self.conv_up = nn.ConvTranspose2d(
            in_channels=out_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn_up = nn.BatchNorm2d(out_channels)



    def forward(self, x):
        x = self.conv_point(x)
        x = self.bn_point(x)
        x = F.relu(x)
        x = self.conv_up(x)
        x = self.bn_up(x)
        x = F.relu(x)
        return x
    
class TrTS_NeXt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,type_norm = "BatchNorm2d",type_activation = "GELU"):
        super(TrTS_NeXt, self).__init__()
        self.out_channels = out_channels

        self.conv_gate = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.Tanh())
        self.out_channels = out_channels

        self.conv_depth = nn.Conv2d(
            in_channels=out_channels,  
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
            groups=out_channels,
        )
        self.LN = nn.GroupNorm(1, out_channels, eps=1e-8) # channel first
        self.conv_point = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()

        self.stride = stride
        self.conv_up = nn.ConvTranspose2d(
            in_channels=out_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_channels
        )


        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "GELU" : 
            self.activation = torch.nn.GELU()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()

    def forward(self, x):
        skip = x[...,self.out_channels:,:,:]
        x = skip*self.conv_gate(x)

        x_in = x
        x = self.conv_depth(x)
        x = self.LN(x)
        x = self.conv_point(x)
        x = self.activation(x)
        x = x_in + x
        
        x = self.conv_up(x)
        x = self.norm_point(x)
        x = self.activation(x)
        return x

    
class TrNext(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,type_norm = "BatchNorm2d",type_activation = "ReLU"):
        super(TrNext, self).__init__()
        self.out_channels = out_channels
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()

        self.conv_up = nn.ConvTranspose2d(
            in_channels=out_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif type_activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif type_activation == "ELU":
            self.activation = torch.nn.ELU()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv_point(x)
        x = self.norm_point(x)
        x = self.conv_up(x)
        #x = F.gelu(x)
        x = self.activation(x)
        return x
    
        
class TrGConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,type_norm = "BatchNorm2d",type_activation = "ReLU"):
        super(TrGConv,self).__init__()
        
        self.c = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups)
        self.m = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups),
            nn.Sigmoid()
            )
        
        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif type_activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif type_activation == "ELU":
            self.activation = torch.nn.ELU()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()
        
    def forward(self, x): 
        m = self.m(x)
        x = self.c(x)
        
        return m*x 


class LastTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,type_norm = "BatchNorm2d",type_activation="ReLU"):
        super(LastTrCNN, self).__init__()
        self.out_channels = out_channels
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()
        self.conv_up = nn.ConvTranspose2d(
            in_channels=out_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if type_activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif type_activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif type_activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif type_activation == "ELU":
            self.activation = torch.nn.ELU()
        elif type_activation == "PReLU":
            self.activation = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv_point(x)
        x = self.norm_point(x)
        x = self.activation(x)
        x = self.conv_up(x)
        return x

class MEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=4, mag_f_dim=3):
        super(MEA, self).__init__()
        self.mag_mask = nn.Conv2d(
            in_channels, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = torch.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim
        self.eps = 1e-12
    
    def forward(self, x, z):
        # x = input stft, x.shape = (B,F,T,2)
        mag = torch.norm(x, dim=-1)
        pha = custom_atan2(x[..., 1], x[..., 0])

        # stage 1
        mag_mask = self.mag_mask(z)
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.relu()
        mag = mag.sum(dim=1)

        # stage 2
        real_mask = self.real_mask(z).squeeze(1)
        imag_mask = self.imag_mask(z).squeeze(1)

        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, self.eps))
        pha_mask = custom_atan2(imag_mask+self.eps, real_mask+self.eps)
        real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)
    
class mapping(nn.Module):
    def __init__(self,in_channels,n_hfft = 513):
        super(mapping, self).__init__()

        ## v1 v2
        #self.conv_re =  nn.Conv2d(in_channels,1,1)
        #self.conv_im =  nn.Conv2d(in_channels,1,1)
        self.act = nn.Tanh()

        ## v3
        #self.conv_re =  nn.Conv2d(in_channels,1,(3,1),padding=(1,0))
        #self.conv_im =  nn.Conv2d(in_channels,1,(3,1),padding=(1,0))

        ## v5 : v3 + linear
        #self.conv_re =  nn.Conv2d(in_channels,1,(3,1),padding=(1,0))
        #self.conv_im =  nn.Conv2d(in_channels,1,(3,1),padding=(1,0))
        #self.fc_re = nn.Linear(n_hfft,n_hfft)
        #self.fc_im = nn.Linear(n_hfft,n_hfft)

        # v7
        self.conv_re =  nn.Conv2d(in_channels,1,(3,3),padding=(1,1))
        self.conv_im =  nn.Conv2d(in_channels,1,(3,3),padding=(1,1))

        self.gconv_re =  nn.Conv2d(in_channels,1,(3,3),padding=(1,1))
        self.gconv_im =  nn.Conv2d(in_channels,1,(3,3),padding=(1,1))

        self.act = nn.Sigmoid()

    def forward(self,x,z) : 
        re = self.conv_re(z)
        im = self.conv_im(z)

        ## v1
        #z = self.act(torch.stack([re,im],dim=-1))
        #z = torch.stack([re,im],dim=-1)

        # v2 ~ v4

        # v5
        # X : [B,F,T]
        #re = torch.permute(re,(0,2,1))
        #im = torch.permute(im,(0,2,1))
        #re = self.fc_re(re)
        #im = self.fc_im(im)
        #re = torch.permute(re,(0,2,1))
        #im = torch.permute(im,(0,2,1))

        # v7
        g_re = self.act(self.gconv_re(z))
        g_im = self.act(self.gconv_im(z))

        re = re * g_re
        im = im * g_im

        re = torch.squeeze(re,dim=1)
        im = torch.squeeze(im,dim=1)


        z = torch.stack([re,im],dim=-1)
        return z

"""
Default configuration from nsh's achitecture.json(2023-01-16)
"""
architecture_orig={
        "encoder": {
            "enc1": {"in_channels": 4, "out_channels": 64, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2]},
            "enc2": {"in_channels": 64, "out_channels": 128, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": 128},
            "enc3": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "groups": 128},
            "enc4": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": 128},
            "enc5": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "groups": 128},
            ## orig
            #"enc6": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": 128}
            ## for FSABlock
            "enc6": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": 64}
        },
        "decoder": {
            ## orig
            #"dec6": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 1]},
            ## for FSABlock
            "dec6": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 1]},
            "dec5": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]},
            "dec4": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
            "dec3": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]},
            "dec2": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
            "dec1": {"in_channels": 128, "out_channels": 4, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]}
        },
        "PE": {"in_channels": 1, "out_channels" : 4 },
        "FGRU": {"in_channels": 128, "hidden_size": 64, "out_channels": 64},
        "FSA": {"in_channels": 64, "hidden_size": 64, "out_channels": 64},
        "TGRU": {"in_channels": 64, "hidden_size": 128, "out_channels": 64, "state_size": 17},
        "MEA": {"in_channels": 4, "mag_f_dim": 3}
    }

class _TRANet_helper(nn.Module):
    def __init__(self,
                n_rfft, 
                architecture=architecture_orig,
                kernel_type = "orig",
                phase_encoder="PE",
                type_TBlock = "TGRU",
                type_FBlock = "FGRU",
                type_CBlock = "None",
                type_out = "MEA",
                T_FGRU = False
                ):
        super(_TRANet_helper, self).__init__()

        self.architecture = architecture
        self.tgru_state_size = self.architecture["TGRU"]["state_size"]
        self.n_rfft = n_rfft

        if kernel_type == "orig" : 
            Encoder = DepthwiseSeparableConv2d
            Decoder = TrCNN
        elif kernel_type == "next" :
            Encoder = DepthwiseNext
            Decoder = TrNext
        elif kernel_type == "GConv" :
            Encoder = GConv
            Decoder = TrGConv
        elif kernel_type == "next_v2" :
            Encoder = TS_NeXt
            Decoder = TrTS_NeXt
        else :
            raise Exception("Unknown kernel_type : {}".format(kernel_type))

        if phase_encoder == "PE" : 
            PE = PhaseEncoderV1
        elif phase_encoder == "PEv2" :
            PE = PhaseEncoderV2
        elif phase_encoder == "PEv3" :
            PE = PhaseEncoderV3
        elif phase_encoder == "PEv4" :
            PE = PhaseEncoderV4
        elif phase_encoder == "PEv5" :
            PE = PhaseEncoderV5
        else :
            PE = phase_encoder

        self.PE = PE(**self.architecture["PE"])
        
        n_enc = len(self.architecture["encoder"])
        self.n_enc = n_enc

        self.enc = []
        module = FirstDownConv2d(**self.architecture["encoder"]["enc1"])
        self.add_module("enc1",module)
        self.enc.append(module)

        for i in range(n_enc-1) : 
            module  = Encoder(**self.architecture["encoder"]["enc{}".format(i+2)])
            self.add_module("enc{}".format(i+2),module)
            self.enc.append(module)

        if type_FBlock == "FGRU":
            self.fgru = FGRUBlock(**self.architecture["FGRU"])
        elif type_FBlock == "FSA" :
            self.fgru = FSABlock(**self.architecture["FSA"])
        elif type_FBlock == "FAT" :
            self.fgru = FATBlock(**self.architecture["FAT"])

        self.type_TBlock = type_TBlock
        if type_TBlock == "TGRU" :
            self.tgru = TGRUBlock(**self.architecture["TGRU"])
        elif type_TBlock == "TLSTM" :
            self.tgru = TLSTMBlock(**self.architecture["TGRU"])
        else :
            raise Exception("Unknown type_TBlock : {}".format(type_TBlock))

        if type_CBlock == "CSA" : 
            self.cblock = CSABlock(**self.architecture["CBlock"])
        else :
            self.cblock = nn.Identity()

        if T_FGRU :
            self.t_fgru = T_FGRUBlock(**self.architecture["T_FGRU"])
        else :
            self.t_fgru = nn.Identity()

        self.dec = []
        for i in range(n_enc-1) : 
            module = Decoder(**self.architecture["decoder"]["dec{}".format(n_enc-i)])
            self.add_module("dec{}".format(n_enc-i),module)
            self.dec.append(module)
        module = LastTrCNN(**self.architecture["decoder"]["dec1"])
        self.add_module("dec1",module)
        self.dec.append(module)

        if type_out == "MEA" : 
            self.out = MEA(**self.architecture["out"])
        elif type_out == "mapping" : 
            self.out = mapping(**self.architecture["out"])


        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)

    def create_dummy_states(self, batch_size, device,num_layers=2):

        pe_state_shape = (2, self.n_rfft, 2, 2)
        if self.type_TBlock == "TGRU" :
            shape = (1, batch_size * self.tgru_state_size, self.tgru.GRU.hidden_size)
            return torch.zeros(*shape).to(device), torch.zeros(*pe_state_shape).to(device)
        elif self.type_TBlock == "TLSTM" :
            shape = (1, batch_size * self.tgru_state_size, self.tgru.GRU.hidden_size)
            return (torch.zeros(*shape).to(device),torch.zeros(*shape).to(device)), torch.zeros(*pe_state_shape).to(device)
        else : 
            raise Exception("Unknown type_TBlock : {}".format(self.type_TBlock))


    def pad_x(self, x, pe_state):
        if is_tracing():
            padded_x = torch.cat((pe_state, x), dim=-2)
            pe_state = padded_x[..., 1:, :]
        else:
            padded_x = F.pad(x, (0, 0, 2, 0), "constant", 0.0)
        return padded_x, pe_state

    def forward(self, x, tgru_state=None, pe_state=None):
        if tgru_state is None: 
            tgru_state,pe_state  = self.create_dummy_states(1,x.device)
        padded_x, pe_state = self.pad_x(x, pe_state)
        # x.size() = [B,F,T,2]
        # feature size is equal to the number of fft bins
        # B:batch_size, T:time_steps, C:channels, F:nfft
        B, M, F, T, _ = padded_x.shape
        x_ = padded_x.permute(0, 1, 4, 2, 3).contiguous()
        abs_x_ = (x_[:,:,[0],:,:]**2 + x_[:,:,[1],:,:]**2)**0.5
        x_ = torch.cat((abs_x_,x_),dim=2)
        B, M, C, F, T = x_.shape
        x_ = x_.view(B,M*C,F,T).contiguous()
        # print(x_.shape)
        # x_.shape == (B,2,F,T)
        x_ = self.PE(x_) # phase encoder would return (B,C,F,T)
        # print(x_.shape)
        x_ = x_.permute(0, 1, 3, 2)  # x_.shape = (B,C,T,F) 
        # in this scope, and between the reshapes, the shape of the data is (B,C,T,F)

        #print("input : {} | {},{}".format(x_.shape,tgru_state.shape,pe_state.shape))
        skip = []
        for i in range(self.n_enc):
            x_ = self.enc[i](x_)
            skip.append(x_)
            #print("enc{} : {}".format(i,x_.shape))


        xf = self.fgru(x_)
     #   print("xf {}".format(xf.shape))
        x_, tgru_state = self.tgru(xf, tgru_state)

        #print("xc {}".format(x_.shape))

        x_ = self.cblock(x_)
#        print("xt {}".format(x_.shape))

        ## only TGRU
        #xt, tgru_state = self.tgru(xd6, tgru_state)

        for i in range(self.n_enc) : 
            x_ = self.dec[i](torch.cat([x_, skip[self.n_enc-i-1]], dim=-3))
            #print("dec{} : {}".format(i,x_.shape))

        z = x_.permute(0, 1, 3, 2)
        # z.shape == (B,C,F,T) 

        direct_stft_reim = self.out(x[:,0], z)
        return direct_stft_reim, tgru_state, pe_state  #direct_stft_reim.shape == (B,F,T,2)

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        #output_folder = output_fp.parent
        #if not os.path.exists(output_folder):
        #    os.makedirs(output_folder)

        dummy_input = torch.randn(2,self.n_rfft, 1, 2).to(device)
        dummy_states, dummy_pe_state  = self.create_dummy_states(1,device)

        torch.onnx.export(
            self,
            (dummy_input, dummy_states, dummy_pe_state),
            output_fp,
            verbose=False,
            opset_version=16,
            input_names=["inputs", "gru_state_in", "pe_state_in"],
            output_names=["outputs", "gru_state_out", "pe_state_out"],
        )

class TRANet_v2(nn.Module):
    def __init__(self, 
        n_fft=1024, 
        n_hop=256,
        architecture = architecture_orig,
        kernel_type = "next",
        phase_encoder = "PE",
        type_FBlock = "FSA",
        type_TBlock = "TGRU",
        type_out = "MEA",
        T_FGRU = False
        ):
        super().__init__()
        self.helper = _TRANet_helper(
            n_fft // 2 + 1,
            architecture = architecture,
            kernel_type= kernel_type,
            phase_encoder=phase_encoder,
            type_FBlock=type_FBlock,
            T_FGRU=T_FGRU,
            type_out = type_out,
            type_TBlock=type_TBlock
            )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.window = torch.hann_window(self.n_fft)

    def _to_spec(self,x_in):

        B,L,M = x_in.shape
        x = x_in.permute(0,2,1).contiguous()
        x = x.view(B*M,L).contiguous()
        if self.n_fft == self.n_hop :
            if L % self.n_fft != 0 :
                x = F.pad(x,(0,self.n_fft - (L % self.n_fft)))
            X = torch.reshape(x,(x.shape[0],self.n_fft,-1))
            X = torch.fft.rfft(X,dim=1)
            X = torch.stack([X.real,X.imag],dim=-1)
        else : 
            X = torch.stft(x, n_fft = self.n_fft, window=torch.hann_window(self.n_fft).to(x.device),return_complex=False)

        if len(x_in.shape) == 3:
            _,F,T,_ = X.shape
            X = X.view(B,M,F,T,-1).contiguous()
            # X = X.permute(0,2,3,1,4).contiguous() # B, F, T, M, C
                        
        return X

    def _to_signal(self, stft,L=None):
        # stft.size() = [B,F,T,2]
        stft = stft[...,0] + 1j * stft[...,1]  # stft.shape (B,F,T)

        if self.n_fft == self.n_hop : 
            out_signal = torch.fft.irfft(stft,dim=1)
            out_signal = torch.reshape(out_signal,(out_signal.shape[0],-1))

        else :
            out_signal = torch.istft(stft, self.n_fft, self.n_hop, self.n_fft, torch.hann_window(self.n_fft).to(stft.device),length=L)

        
        return out_signal  # out_signal.shape == (B,N), N = num_samples
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        B, L, M = x.shape
        X = self._to_spec(x)
        # x.shape == (B,F,T,2)
        input_tgru_state, pe_state = self.helper.create_dummy_states(X.shape[0], X.device)
        direct_stft_reim, _, _ = self.helper(X, input_tgru_state, pe_state)

        direct_out = self._to_signal(direct_stft_reim,L)
        # reverberant_out = self._mask2signal(in_mag, in_phase, mask_mag_r, mask_phase_r)

        
        return direct_out[:, :L]

    def enhance_speech(self, x, _aux):
        return self.forward(x)[0].detach().cpu().numpy()

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        self.helper.to_onnx(output_fp, device)


def test(
    architecture = architecture_orig,
    n_fft = 512,
    n_hop = 128
         ) : 
    batch_size = 2
    m = TRANet(
        architecture=architecture,
        n_fft=n_fft,
        n_hop=n_hop,
        type_FBlock="FSA"
        )
    inputs = torch.randn(batch_size,16000)
    y = m(inputs)
    print(y.shape)
    
    
if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import soundfile as sf
    import os
    import yaml
    
    with open("train_TRANet_DNS_TRULoss_app_ver2_2.yaml", 'r') as f:
        config_dict = yaml.full_load(f)
    # config_dict = yaml.full_load(args.config)
    nnet_conf = config_dict["model"]
    nnet = TRANet_v2(**nnet_conf)
    checkpoint_dict = torch.load('epoch.77.pkl', map_location='cpu')
    nnet.load_state_dict(checkpoint_dict['model_state_dict'])
    nnet.eval()
    nnet.to_onnx("TRANetV2")

# 

def parse_yaml(yaml_conf):
    if not os.path.exists(yaml_conf):
        raise FileNotFoundError(
            "Could not find configure files...{}".format(yaml_conf))

    batch_size = config_dict["dataloader"]["batch_size"]
    if batch_size <= 0:
        raise ValueError("Invalid batch_size: {}".format(batch_size))
    return config_dict