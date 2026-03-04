import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpSE._coders import get_coder
from mpSE._necks import get_neck
from mpSE._edges import get_edge 
from mpSE._skips import get_skip
from mpSE._filter import get_filter

default_architecture = {
    "MIC_SCALE" : 4.0,
    "type_PE" : "PhaseEncoderV1",
    "type_encoder":"DepthwiseSeparableConv2d",
    "type_decoder":"TrDepthwiseSeparableConv2d",
    "type_skip":"SkipCat",
    "type_mask" : "Identity2arg1out",
    "type_FBlock": "FGRUBlock",
    "type_CBlock" : "NoneBlock",
    "type_TBlock": "TGRUBlock",
    "type_mask" : "MEA",
    "encoder": {
        "enc1": {"in_channels": 4, "out_channels": 64, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2]},
        "enc2": {"in_channels": 64, "out_channels": 128, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": 128},
        "enc3": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "groups":128},
        "enc4": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": 128},
        "enc5": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "groups": 128},
        ## FSABlock
        "enc6": {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": 128}
    },
    "decoder": {
        ## FSABlock
        "dec6": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 1]},
        "dec5": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]},
        "dec4": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
        "dec3": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]},
        "dec2": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
        "dec1": {"in_channels": 128, "out_channels": 4, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2]}
    },
    "filter" : {"in_channels":128,"hidden_size":17,"out_size":257,"n_erb":40},
    "skipper":{
      "skip6":{},
      "skip5":{},
      "skip4":{},
      "skip3":{},
      "skip2":{},
      "skip1":{}
    },
    "PE": {"in_channels": 1, "out_channels" : 4, "type_norm" : "BatchNorm2d"},
    "FBlock": {"in_channels": 128, "hidden_size": 64, "out_channels": 64},
    "TBlock": {"in_channels": 64, "hidden_size": 128, "out_channels": 64, "state_size": 17, "num_layers": 1},
    "CBlock" : {},
    "Mask": {"in_channels": 4, "mag_f_dim": 3}
}

class mpNC(nn.Module):
    def __init__(self,architecture=default_architecture):
        super(mpNC, self).__init__()

        self.architecture = architecture

        ###  Types 
        Encoder = get_coder(self.architecture["type_encoder"])
        Decoder = get_coder(self.architecture["type_decoder"])
        PE = get_edge(self.architecture["type_PE"]) 
        FBlock= get_neck(self.architecture["type_FBlock"])
        TBlock = get_neck(self.architecture["type_TBlock"])
        CBlock = get_neck(self.architecture["type_CBlock"])
        Skip = get_skip(self.architecture["type_skip"])
        Mask = get_edge(self.architecture["type_mask"])
        Filter = get_filter(self.architecture["type_filter"])

        self.MIC_SCALE = self.architecture["MIC_SCALE"]

        ###  Instanciation
        if self.architecture["CR"]["use"] : 
            self.cr = get_edge("CR")(**self.architecture["CR"])
            self.icr = get_edge("iCR")(**self.architecture["CR"])
        else :
            self.cr = nn.Identity()
            self.icr = nn.Identity()

        self.PE = PE(**self.architecture["PE"])

        n_enc = len(self.architecture["encoder"])
        self.n_enc = n_enc

        self.enc = []
        for i in range(n_enc) : 
            module  = Encoder(**self.architecture["encoder"]["enc{}".format(i+1)])
            self.add_module("enc{}".format(i+2),module)
            self.enc.append(module)


        self.dec = []
        for i in range(n_enc) : 
            module = Decoder(**self.architecture["decoder"]["dec{}".format(n_enc-i)])
            self.add_module("dec{}".format(n_enc-i),module)
            self.dec.append(module)

        self.filter = Filter(**self.architecture["filter"])

        self.fblock = FBlock(**self.architecture["FBlock"])
        self.tblock = TBlock(**self.architecture["TBlock"])
        self.cblock = CBlock(**self.architecture["CBlock"])

        self.skipper =[]
        for i in range(n_enc):
            module = Skip(**self.architecture["skipper"]["skip{}".format(n_enc-i)])
            self.add_module("skip{}".format(n_enc-i),module)
            self.skipper.append(module)

        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)
        self.skipper = nn.ModuleList(self.skipper)
        self.mask = Mask(**self.architecture["Mask"])

    def create_dummy_states(self, batch_size, device, **kwargs):
        state_p_shape = (1, 257, 2, 2)
        shape = (self.architecture["TBlock"]["num_layers"],
                 batch_size * self.architecture["TBlock"]["state_size"], self.architecture["TBlock"]["hidden_size"])
        return torch.zeros(*shape).to(device), torch.zeros(*state_p_shape).to(device)

    def pad_x(self, x, state_p, n_time = 2):
        if torch._C._is_tracing() : 
            padded_x = torch.cat((state_p, x), dim=-2)
            state_p = padded_x[..., 1:, :]
        else:
            padded_x = F.pad(x, (0, 0, n_time-1, 0), "constant", 0.0)
        return padded_x, state_p

    def forward(self, x, state_t=None, state_p=None):
        """
        x : (B,F,T,2), input stft (real, imag)
        state_t : (num_layers, B*state_size, hidden_size), temporal block state
        state_p : (1,F,2,2), input buffer
        """

        x = x * self.MIC_SCALE
    
        if state_t is None:
            state_t,state_p  = self.create_dummy_states(1,x.device)

        padded_x, state_p = self.pad_x(x, state_p, 3)
        x_ = padded_x.permute(0, 3, 1, 2)
        x_ = self.PE(x_) # phase encoder would return (B,C,F,T)
        x_ = self.cr(x_)
        x_ = x_.permute(0, 1, 3, 2)  # x_.shape = (B,C,T,F) 
        skip = []
        for i in range(self.n_enc):
            x_ = self.enc[i](x_)
            skip.append(x_)
            #print(f"Encoder {i} output shape : {x_.shape}")

        xf = self.fblock(x_)
        xc = self.cblock(xf)
        x_, state_t = self.tblock(xc, state_t)
        x = self.filter(x_,x)

        for i in range(self.n_enc) : 
            x_ = self.dec[i](self.skipper[i](x_,skip[self.n_enc-i-1]))
            #print(f"Decoder {i} output shape : {x_.shape}")
        z = x_.permute(0, 1, 3, 2)

        z = self.icr(z)
        y = self.mask(x, z)

        y = y / self.MIC_SCALE
        #exit()
        return y, state_t, state_p#direct_stft_reim.shape == (B,F,T,2)

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        dummy_input = torch.randn(1,257, 1, 2).to(device)
        dummy_states, dummy_state_p  = self.create_dummy_states(1,device)

        torch.onnx.export(
            self,
            (dummy_input, dummy_states, dummy_state_p),
            output_fp,
            verbose=False,
            dynamo=False,
            opset_version=16,
            input_names=["inputs", "gru_state_in", "pe_state_in"],
            output_names=["outputs", "gru_state_out", "pe_state_out"],
        )

class mpNC_helper(nn.Module):
    def __init__(self,hp = None):
        super().__init__()
        if hp is None :
            self.model = mpNC()
        else : 
            self.model = mpNC(architecture=hp.model.architecture)
        self.frame_size = 512
        self.hop_size = 128
        PI = 3.14159265358979323846
        self.window = torch.zeros(self.frame_size)
        for i in range(self.frame_size) : 
            self.window[i] = torch.tensor(0.5 * (1.0 - np.cos(2.0 * PI * i / self.frame_size)) / np.sqrt((self.frame_size* 0.3750) / self.hop_size))

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
        input_state_t, state_p = self.model.create_dummy_states(X.shape[0], X.device)

        #print(f"input h {input_state_t.shape} | e {state_p.shape}")

        Y, h, e = self.model(X, input_state_t, state_p)
        #print(f"output h {h.shape} | e {e.shape}")

        y = self._to_signal(Y)
        return y[:, :L]

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        self.model.to_onnx(output_fp, device)

    
def test_mpNC():
    model = mpNC_helper()
    model.eval()

    x = torch.rand(2,49152)
    y = model(x)
    print(y.shape)


    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    
    print(f"MACs : {macs}, #Params : {params}")