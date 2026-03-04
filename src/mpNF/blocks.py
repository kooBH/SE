import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from mpNF.utils import *
from mpNF.modules import *

print_ENABLED = False

def print_mpNF(*args, **kwargs):
    if print_ENABLED:
        print(*args, **kwargs)

class ERBEncoder(nn.Module):
    def __init__(
        self, 
        arch
        # orig -> depthwise separable conv
        ):
        super(ERBEncoder, self).__init__()

        type_encoder = arch["type_encoder"]

        Encoder = get_module(type_encoder)

        self.n_causal = arch["n_causal"]
        self.infer = False
        if torch._C._is_tracing():
            self.infer = True
        self.n_enc = arch["n_enc"]
        self.enc = []

        self.skipper = []

        for i in range(self.n_enc) : 
            module = Encoder(**arch[f"enc{i+1}"])
            self.enc.append(module)
            self.add_module(f"enc{i+1}",module)

        self.enc = nn.ModuleList(self.enc)

    def temporal(self,x):
        if self.infer : 
            # TODO
            pass
        else : 
            # x : [B,C,T,F]
            x = F.pad(x,(0,0,self.n_causal,0),"constant",0.0)
        return x

    def forward(self,x):
        # x : [B,C,T,F]
        print_mpNF(f"ERBEncoder:: {x.shape} ->",end="")
        x = self.temporal(x)
        print_mpNF(f" {x.shape}")

        skip = []
        for i in range(self.n_enc) : 
            print_mpNF(f"ERBEncoder::enc[{i}] : {x.shape} ->",end="")
            x = self.enc[i](x)
            print_mpNF(f"{x.shape}")
            skip.append(x)

        # flatten C*F -> emb
        # x : [B,T,C*F//4]
        #x = x.permute(0,2,3,1)
        #x = x.reshape(x.shape[0],x.shape[1],-1)

        return x, skip

# DeepfilterNet Df Encoder
class DFEncoder(nn.Module):
    def __init__(
        self, 
        arch
        # orig -> depthwise separable conv
        ):
        super(DFEncoder, self).__init__()
        type_encoder = arch["type_encoder"]

        Encoder = get_module(type_encoder)

        self.n_causal = arch["n_causal"]
        self.infer = False
        if torch._C._is_tracing():
            self.infer = True
        self.n_enc = arch["n_enc"]
        self.enc = []

        self.skipper = []

        for i in range(self.n_enc) : 
            module = Encoder(**arch[f"enc{i+1}"])
            self.enc.append(module)
            self.add_module(f"enc{i+1}",module)

        self.enc = nn.ModuleList(self.enc)

    def temporal(self,x):
        if self.infer : 
            # TODO
            pass
        else : 
            # x : [B,C,T,F]
            x = F.pad(x,(0,0,self.n_causal,0),"constant",0.0)
        return x

    def forward(self,x):
        # x : [B,C,T,F]
        print_mpNF(f"DFEncoder:: {x.shape} ->",end="")
        x = self.temporal(x)
        print_mpNF(f" {x.shape}")

        skip = []
        for i in range(self.n_enc) : 
            print_mpNF(f"DFEncoder::enc[{i}] : {x.shape} ->",end="")
            x = self.enc[i](x)
            print_mpNF(f"{x.shape}",)
            skip.append(x)
        # flatten C*F -> emb

        # x : [B,T,emb]
        print_mpNF(f"DFEncoder::{x.shape}")

        return x, skip
    
class EncoderFusion(nn.Module):
    def __init__(self,arch):
        super(EncoderFusion, self).__init__()

        if arch["type_emb"] == "linear" : 
            self.pre_emb_df = Flatter()
            self.pre_emb_erb = Flatter()
            self.layer_emb_df = GroupedLinear(**arch["linear_df"])
            self.layer_emb_erb = nn.Identity()
        else : 
            self.pre_emb_df = nn.Identity()
            self.pre_emb_erb = nn.Identity()
            self.layer_emb_df = nn.Identity()
            self.layer_emb_erb = nn.Identity()


        if arch["type_fusion"] == "cat" : 
            self.emb = EmbCat()
        elif arch["type_fusion"] == "FC" : 
            self.emb = EmbFC(arch["rnn"]["input_size"])
        elif arch["type_fusion"] == "FSA" :
            self.emb = EmbFSA(**arch["FSA"])
        else : 
            raise Exception(f"ERROR::Unknown encoder fusion type : {arch['type']}")
        self.rnn = SqueezedGRU(**arch["rnn"])
        self.lsnr = nn.Linear(**arch["lsnr"])

    def forward(self,x_erb,x_df,hidden_state=None):
        print_mpNF(f"EncoderFusion::pre {x_erb.shape}, {x_df.shape}")
        x_df = self.pre_emb_df(x_df)
        x_erb = self.pre_emb_erb(x_erb)
        print_mpNF(f" -> {x_erb.shape}, {x_df.shape}")

        print_mpNF(f"EncoderFusion::emb {x_erb.shape}, {x_df.shape}")
        x_erb = self.layer_emb_erb(x_erb)
        x_df = self.layer_emb_df(x_df)
        print_mpNF(f" -> {x_erb.shape}, {x_df.shape}")

        print_mpNF(f"EncoderFusion::fusion {x_erb.shape} {x_df.shape} {hidden_state.shape}")
        emb = self.emb(x_erb,x_df)
        print_mpNF(f"EncoderFusion:: fused {emb.shape}")
        emb, hidden_state = self.rnn(emb,hidden_state)
        print_mpNF(f"EncoderFusion:: {emb.shape}")
        lsnr = self.lsnr(emb)

        return emb, hidden_state, lsnr

class EncoderFusionTF(nn.Module):
    def __init__(self,arch):
        super(EncoderFusionTF, self).__init__()

        if arch["type_emb"] == "linear" : 
            self.pre_emb_dfh = Flatter()
            self.pre_emb_dfl = Flatter()
            self.pre_emb_erb = Flatter()
            self.layer_emb_dfh = GroupedLinear(**arch["linear_dfh"])
            self.layer_emb_dfl = GroupedLinear(**arch["linear_dfl"])
            self.layer_emb_erb = nn.Identity()
        else : 
            self.pre_emb_dfh = nn.Identity()
            self.pre_emb_dfl = nn.Identity()
            self.pre_emb_erb = nn.Identity()
            self.layer_emb_dfh = nn.Identity()
            self.layer_emb_dfl = nn.Identity()
            self.layer_emb_erb = nn.Identity()


        if arch["type_fusion"] == "cat" : 
            self.emb = EmbCat()
        elif arch["type_fusion"] == "FC" : 
            self.emb = EmbFC(arch["rnn"]["input_size"])
        elif arch["type_fusion"] == "FSA" :
            self.emb = EmbFSA(**arch["FSA"])
        else : 
            raise Exception(f"ERROR::Unknown encoder fusion type : {arch['type']}")
        self.rnn = SqueezedGRU(**arch["rnn"])

    def forward(self,x_erb,x_dfh, x_dfl, hidden_state=None):
        print_mpNF(f"EncoderFusion::pre {x_erb.shape}, {x_dfh.shape}, {x_dfl.shape}")
        x_erb = self.pre_emb_erb(x_erb)
        x_dfh = self.pre_emb_dfh(x_dfh)
        x_dfl = self.pre_emb_dfl(x_dfl)
        print_mpNF(f" -> {x_erb.shape}, {x_dfh.shape}, {x_dfl.shape}")

        print_mpNF(f"EncoderFusion::emb {x_erb.shape}, {x_dfh.shape}, {x_dfl.shape}")
        x_erb = self.layer_emb_erb(x_erb)
        x_dfh = self.layer_emb_dfh(x_dfh)
        x_dfl = self.layer_emb_dfl(x_dfl)
        print_mpNF(f" -> {x_erb.shape}, {x_dfh.shape}, {x_dfl.shape}")

        print_mpNF(f"EncoderFusion::fusion {x_erb.shape} {x_dfh.shape} {x_dfl.shape} | {hidden_state.shape}")
        emb = self.emb(x_erb,x_dfh,x_dfl)
        print_mpNF(f"EncoderFusion:: fused {emb.shape}")
        emb, hidden_state = self.rnn(emb,hidden_state)
        print_mpNF(f"EncoderFusion:: {emb.shape}")

        return emb, hidden_state

class ERBDecoder(nn.Module):
    def __init__(self,arch,
        type_decoder="DFNTrConvLayer"
        ):
        super(ERBDecoder, self).__init__()

        self.rnn = SqueezedGRU(**arch["rnn"])
        Decoder = get_module(type_decoder)

        self.n_dec = arch["n_dec"]
        self.n_channels = arch["pdec1"]["out_channels"]

        self.pdec = []
        self.skip = []
        self.dec = []
        for i in range(self.n_dec) : 
            module = nn.Conv2d(**arch[f"pdec{i+1}"])
            self.pdec.append(module)
            self.add_module(f"pdec{i+1}",module)

            module = EmbAdd(**arch[f"skip{i+1}"])
            self.skip.append(module)
            self.add_module(f"skip{i+1}",module)

            module = Decoder(**arch[f"dec{i+1}"])
            self.dec.append(module)
            self.add_module(f"dec{i+1}",module)

        self.dec = nn.ModuleList(self.dec)
        self.skip = nn.ModuleList(self.skip)
        self.pdec = nn.ModuleList(self.pdec)

    def forward(self,x,skips,h=None):
        x,h = self.rnn(x,h)

        # x : [B,T,F] -> [B,C,T,F/C]
        print_mpNF(f"ERBDecoder:: {x.shape} ->",end=" ")
        B,T,F = x.shape
        x = x.view(B,T,F//self.n_channels,self.n_channels)
        x = x.permute(0,3,1,2)
        print_mpNF(f"{x.shape}")

        for i in range(self.n_dec) : 
            print_mpNF(f"skips[{i}]: {skips[i].shape}")
        for i in range(self.n_dec) : 
            s = self.pdec[i](skips[-i-1])
            print_mpNF(f"ERBDecoder::dec[{i+1}] : {skips[-i-1].shape} -> s : {s.shape} + x : {x.shape}")
            x = self.skip[i](x,s)
            x = self.dec[i](x)
        print_mpNF(f"ERBDecoder:: {x.shape}")

        return x,h

class DFDecoder(nn.Module):
    def __init__(self,
        arch):
        super(DFDecoder, self).__init__()

        self.pconv = DFNConvLayer(**arch["pconv"])
        self.rnn = SqueezedGRU(**arch["rnn"])
        self.df_out = GroupedLinear(**arch["linear"])
        self.fc_alpha = nn.Linear(**arch["alpha"])
        self.n_channels = arch["pconv"]["out_channels"]

    def forward(self,x,skip, h= None):
        x,h = self.rnn(x,h)

        skip = self.pconv(skip[0])
        alpha = self.fc_alpha(x)
        x = self.df_out(x) # groupedlinear

        print_mpNF(f"DFDecoder:: {x.shape} {skip.shape}")

        # skip : [B,C,T,F] -> [B,T,F,C]
        skip = skip.permute(0,2,3,1)

        # x : [B,T,F] -> [B,C,T,F/C]
        B,T,F = x.shape
        x = x.view(B,T,F//self.n_channels,self.n_channels)

        x = x + skip

        return x,h,alpha
    
class ErbMask(nn.Module):
    def __init__(self,
        erb_inv,
        post_filter=False,
        beta = 0.2):
        super(ErbMask, self).__init__()

        self.register_buffer('erb_inv', erb_inv)
        self.eps = 1e-12

        self.post_filter = post_filter
        self.beta = beta

    def pf(self, mask: torch.Tensor) -> torch.Tensor:
        """Post-Filter proposed by Valin et al. [1].
        Args:
            mask (Tensor): Real valued mask, typically of shape [B, C, T, F].
        Refs:
            [1]: Valin et al.: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech.
        """
        mask_sin = mask * torch.sin(np.pi * mask / 2)
        mask_pf = (1 + self.beta) * mask / (1 + self.beta * mask.div(mask_sin.clamp_min(self.eps)).pow(2))
        return mask_pf

    def forward(self,spec, mask):
        # spec : [B,1,T,F_df]
        # mask : [B,1,T,F_erb]
        print_mpNF(f"ErbMask:: 1 {spec.shape} {mask.shape}")

        # [B,1,T,F]
        mask  = mask.matmul(self.erb_inv)

        if self.post_filter:
            mask = self.pf(mask)
        mask = mask.squeeze(1)

        print_mpNF(f"ErbMask:: 2 {spec.shape} {mask.shape}")
        
        return spec*mask

class DfOp(nn.Module):
    def __init__(self, n_df, base_bin=0, n_order=5, infer=False):
        super(DfOp, self).__init__()
        self.n_df = n_df
        self.n_order = n_order
        self.base_bin = base_bin

        if torch._C._is_tracing():
            self.infer = True
        else :
            self.infer = infer

        self.pad = nn.ConstantPad2d((0, 0, n_order - 1, 0), 0.0)

        # buffer for inference : [B,C,T,F,N-1]
        # B = C = T = 1 for frame by frame mono input
        self.df_buffer = torch.zeros(1,1,1,n_df,n_order-1)

    def temporal(self,x):
        # Buffering
        if self.infer :
            # x : [1,1,F]
            # -> unfolded_x : [1,1,F,N]
            y = x.unsqueeze(-1)
            y = torch.cat((self.df_buffer,y),dim=-1)
        # unfolding
        else :
            # x : [B,T,F]
            # -> padded x : [B,T,F,N]
            y = self.pad(x).unfold(1,self.n_order,1)
        return y

    def forward(self,spec,coef):
        # spec : [B,T,F]
        # coef : [B,T,F_df,2*N(complex)]
        print_mpNF(f"DfOp::spec {spec.shape} | coef {coef.shape}")

        B,T,F_df,N = coef.shape

        # spec_df : [B,T,F_df]
        spec_df = spec.narrow(2,self.base_bin,F_df)
        # unfolded_spec : [B,T,F_df,N]
        unfolded_spec = self.temporal(spec_df)
        print_mpNF(f"DfOp::unfoled {unfolded_spec.shape}")

        coef = coef[...,:self.n_order] + coef[...,self.n_order:]*1j

        # DfOp
        # [B,T,F_df,N] * [B,T,F_df,N] -> [B,T,F_df]
        filtered = torch.sum(unfolded_spec * coef, dim=-1)
        spec[:,:,self.base_bin:self.base_bin+self.n_df] = filtered
        return spec

class InterMask(nn.Module) : 
    def __init__(self,arch):
        super(InterMask, self).__init__()

        type_cross = arch["type_cross"]

        if type_cross == "None" :
            self.post = Identity2arg()
        elif type_cross == "Linear" :
            self.post = InterMaskLinear(**arch["Linear"])
        else :
            raise Exception(f"ERROR::Unknown cross mask type : {type_cross}")
        
    def forward(self,m_erb,m_df):
        # m_erb : [B,T,F_erb]
        # m_df : [B,T,F_df]
        print_mpNF(f"InterMask:: {m_erb.shape} {m_df.shape}")
        m_erb,m_df = self.post(m_erb,m_df)
        return m_erb,m_df

class InterMaskTF(nn.Module) : 
    def __init__(self,arch):
        super(InterMaskTF, self).__init__()

        type_cross = arch["type_cross"]

        if type_cross == "None" :
            self.post = Identity3arg()
        elif type_cross == "Linear" :
            self.post = InterMaskLinear(**arch["Linear"])
        else :
            raise Exception(f"ERROR::Unknown cross mask type : {type_cross}")
        
    def forward(self,m_erb,m_dfh, m_dfl):
        # m_erb : [B,T,F_erb]
        # m_df : [B,T,F_df]
        print_mpNF(f"InterMask:: {m_erb.shape} {m_dfh.shape} {m_dfl.shape}")
        m_erb,m_dfh,m_dfl = self.post(m_erb,m_dfh, m_dfl)
        return m_erb,m_dfh,m_dfl


