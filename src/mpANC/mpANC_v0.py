"""
  type_FBlock : "FSA3"
  type_TBlock : "TGRU"
  type_CBlcok : "None"
  kernel_type : next #orig, next, GConv
  skipGRU : False
  phase_encoder : "PE"

"""
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
    def __init__(self, in_channels=1, out_channels = 4,type_norm="None", alpha=0.5,bias=True):
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

class FirstDownConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(FirstDownConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x

class LastTrCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(LastTrCNN, self).__init__()
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

    def forward(self, x):
        x = self.conv_point(x)
        x = self.bn_point(x)
        x = F.relu(x)
        x = self.conv_up(x)
        return x

class DepthwiseNext(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1,type_norm = "BatchNorm2d",type_activation="ReLU"):
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
            self.norm_depth= nn.BatchNorm2d(in_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_depth= nn.InstanceNorm2d(in_channels,track_running_stats=True)
        elif type_norm == "InstanceNorm" : 
            self.norm_depth= nn.GroupNorm(in_channels,in_channels)
        elif type_norm == "LayerNorm" : 
            self.norm_depth= nn.GroupNorm(1,in_channels)
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

class TrNext(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,groups=1,type_norm = "BatchNorm2d",type_activation = "ReLU",output_padding=0):
        super(TrNext, self).__init__()
        self.out_channels = out_channels
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(in_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(in_channels,track_running_stats=True)
        else :
            self.norm = nn.Identity()

        self.conv_up = nn.ConvTranspose2d(
            in_channels=in_channels,  # because it passed already in the previous conv
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            output_padding=output_padding
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
        x = self.conv_up(x)
        x = self.norm(x)
        x = self.conv_point(x)
        #x = F.gelu(x)
        x = self.activation(x)
        return x

class FSA3Block(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(FSA3Block,self).__init__()

        self.pc = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))
        self.bnc = nn.BatchNorm2d(in_channels)
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bnsa = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape

        yc = self.pc(x)
        yc = self.bnc(yc)

        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        ysa,h = self.SA(x_,x_,x_)
        
        ysa = ysa.reshape(B, T, F, C)
        ysa = ysa.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        ysa = self.bnsa(ysa)
        ysa = self.relu(ysa)

        output = yc + ysa

        return output

# Subband TGRU with residual
class TGRUBlock2(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels,num_layers=1,**kwargs):
        super(TGRUBlock2, self).__init__()

        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True, num_layers=num_layers)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        x1 = x.permute(0, 3, 2, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C)

        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)

        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 3, 2, 1)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output) + x
        return output, rnn_state

class SkipCat(nn.Module):
    def __init__(self, **kwargs):
        super(SkipCat,self).__init__()
    
    def forward(self, x,skip,dim=-3):
        # Expected dim : [B,C,T,F]
        # -> [B,C1+C2,T,F]
        return torch.cat([x,skip],dim=dim)

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


class MEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=4, mag_f_dim=3, activation = "ReLU",n_dim=257,**kwargs):
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

        if activation == "ReLU" : 
            self.mag_act = nn.ReLU()
        elif activation == "LSigmoid" : 
            self.mag_act = LSigmoid(n_dim)
        else :
            raise Exception("ERROR::MEA::Unsupported activation : {}".format(activation))
    
    def forward(self, x, z):
        # x = input stft, x.shape = (B,F,T,2)
        mag = torch.norm(x, dim=-1)
        pha = custom_atan2(x[..., 1], x[..., 0])

        # stage 1
        mag_mask = self.mag_mask(z)
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * self.mag_act(mag_mask)
        mag = mag.sum(dim=1)

        # stage 2
        real_mask = self.real_mask(z).squeeze(1)
        imag_mask = self.imag_mask(z).squeeze(1)

        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, self.eps))
        pha_mask = custom_atan2(imag_mask+self.eps, real_mask+self.eps)
        real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)

architecture = {
    "encoder": {
        "enc1": {"in_channels": 4, "out_channels": 64, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2]},
        "enc2": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": 64},
        "enc3": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "groups":64},
        "enc4": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": 64},
        "enc5": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "groups": 64},
        ## FSABlock
        "enc6": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": 32}
    },
    "decoder": {
        ## FSABlock
        "dec6": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 1]},
        "dec5": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]},
        "dec4": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
        "dec3": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]},
        "dec2": {"in_channels": 128, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
        "dec1": {"in_channels": 128, "out_channels": 4, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]}
    },
    "skipper":{
      "skip6":{},
      "skip5":{},
      "skip4":{},
      "skip3":{},
      "skip2":{},
      "skip1":{}
    },
    "PE": {"in_channels": 1, "out_channels" : 4, "type_norm" : "BatchNorm2d"},
    "FGRU": {"in_channels": 64, "hidden_size": 64, "out_channels": 64},
    "FSA": {"in_channels": 64, "hidden_size": 8, "out_channels": 64},
    "TGRU": {"in_channels": 64, "hidden_size": 64, "out_channels": 64, "state_size": 17},
    "MEA": {"in_channels": 4, "mag_f_dim": 3}
}

class mpANC_v0(nn.Module):
    def __init__(self):
        super(mpANC_v0, self).__init__()

        self.architecture = architecture
        self.tgru_state_size = 17
        self.n_rfft = 257

        Encoder = DepthwiseNext
        Decoder = TrNext

        PE = PhaseEncoderV1

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

        self.fgru = FSA3Block(**self.architecture["FSA"])

        self.tgru = TGRUBlock2(**self.architecture["TGRU"])

        self.dec = []
        for i in range(n_enc-1) : 
            module = Decoder(**self.architecture["decoder"]["dec{}".format(n_enc-i)])
            self.add_module("dec{}".format(n_enc-i),module)
            self.dec.append(module)
        module = LastTrCNN(**self.architecture["decoder"]["dec1"])
        self.add_module("dec1",module)
        self.dec.append(module)

        skipper = SkipCat
        self.skipper =[]
        for i in range(n_enc):
            module = skipper(**self.architecture["skipper"]["skip{}".format(n_enc-i)])
            self.add_module("skip{}".format(n_enc-i),module)
            self.skipper.append(module)

        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)
        self.skipper = nn.ModuleList(self.skipper)
        self.mea = MEA(**self.architecture["MEA"])

    def create_dummy_states(self, batch_size, device, **kwargs):
        pe_state_shape = (1, self.n_rfft, 2, 2)
        shape = (self.tgru.GRU.num_layers, batch_size * self.tgru_state_size, self.tgru.GRU.hidden_size)
        return torch.zeros(*shape).to(device), torch.zeros(*pe_state_shape).to(device)

    def pad_x(self, x, pe_state, n_time = 2):
        if torch._C._is_tracing() :
            padded_x = torch.cat((pe_state, x), dim=-2)
            pe_state = padded_x[..., 1:, :]
        else:
            padded_x = F.pad(x, (0, 0, n_time-1, 0), "constant", 0.0)
        return padded_x, pe_state

    def forward(self, x, tgru_state=None, pe_state=None):
        if tgru_state is None:
            tgru_state,pe_state  = self.create_dummy_states(1,x.device)
        padded_x, pe_state = self.pad_x(x, pe_state, 3)
        x_ = padded_x.permute(0, 3, 1, 2)
        x_ = self.PE(x_) # phase encoder would return (B,C,F,T)
        x_ = x_.permute(0, 1, 3, 2)  # x_.shape = (B,C,T,F) 
        skip = []
        for i in range(self.n_enc):
            x_ = self.enc[i](x_)
            skip.append(x_)

        xf = self.fgru(x_)
        x_, tgru_state = self.tgru(xf, tgru_state)

        for i in range(self.n_enc) : 
            x_ = self.dec[i](self.skipper[i](x_,skip[self.n_enc-i-1]))
        z = x_.permute(0, 1, 3, 2)

        direct_stft_reim = self.mea(x, z)
        return direct_stft_reim, tgru_state, pe_state  #direct_stft_reim.shape == (B,F,T,2)

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        dummy_input = torch.randn(1,self.n_rfft, 1, 2).to(device)
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

class mpANC_v0_helper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mpANC_v0()
        self.frame_size = 512
        self.hop_size = 128
        self.window = torch.hann_window(self.frame_size)

    def _to_spec(self,x):
        B,L = x.shape
        if self.frame_size == self.hop_size :

            if L % self.frame_size != 0 :
                x = F.pad(x,(0,self.frame_size - (L % self.frame_size)))
            X = torch.reshape(x,(x.shape[0],self.frame_size,-1))
            X = torch.fft.rfft(X,dim=1)
            X = torch.stack([X.real,X.imag],dim=-1)
        else : 
            X = torch.stft(x, n_fft = self.frame_size, hop_length = self.hop_size, window=self.window.to(x.device),return_complex=False)

        return X


    def _to_signal(self, stft):
        stft = stft[...,0] + 1j * stft[...,1]  

        if self.frame_size == self.hop_size : 
            out_signal = torch.fft.irfft(stft,dim=1)
            out_signal = torch.reshape(out_signal,(out_signal.shape[0],-1))
        else :
            out_signal = torch.istft(stft, self.frame_size, self.hop_size, self.frame_size, self.window.to(stft.device))
        
        return out_signal 

    def forward(self, x):
        B,L = x.shape
        X = self._to_spec(x)
        input_tgru_state, pe_state = self.model.create_dummy_states(X.shape[0], X.device)

        Y, _, _ = self.model(X, input_tgru_state, pe_state)

        y = self._to_signal(Y)
        return y[:, :L]

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        self.model.to_onnx(output_fp, device)


if __name__ == "__main__" :

    import librosa as rs
    import soundfile as sf

    m = mpANC_v0_helper()
    # need to train first
    #m.load_state_dict(torch.load('mpANC_v0.pt'))

    x = rs.load("TV_16kHz_mono.wav",sr=16000)[0]

    x = torch.tensor(x).unsqueeze(0)

    y = m(x)
    y = y.squeeze(0).detach().numpy()

    sf.write("output.wav",y,16000)

    