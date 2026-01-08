import torch
import torch.nn as nn
import torch.nn.functional as F

from mpNF.utils import *
from mpNF.blocks import *
from mpNF.modules import *
from mpNF.features import FeatureCalculator

arch_orig = {
    "params":{
        "nb_erb" : 24,
        "nb_df" : 160,
        "n_causal" : 2,
        "n_order" : 5
    },
    "erb_encoder": {
        "n_causal" : 2,
        "n_enc" : 4,
        # First kernel (1 + n_causal,3)
        "enc1": {"in_channels": 1, "out_channels": 64, "kernel_size": [3, 3], "stride": [1,1], "padding": [0,1],"groups" : -1,"bias" : False},
        "enc2": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": -1,"bias" :False },
        "enc3": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups":-1, "bias" : False},
        "enc4": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,1], "padding": [0,1], "groups": -1, "bias" : False}
    },
    "df_encoder": {
        "n_causal" : 2,
        "n_enc" : 2,
        # First kernel (1 + n_causal,3)
        "enc1": {"in_channels": 2, "out_channels": 64, "kernel_size": [3, 3], "stride": [1,1], "padding": [0,1], "groups": -1, "bias" : False},
        "enc2": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": -1, "bias" : False},
        # input_size = conv_ch * nb_df//2
        # hidden_size = conv_ch * nb_erb//4
        "linear" : {"input_size" : 5120 , "hidden_size": 320 ,"groups" :8 }
    },
    "encoder_fusion": {
        "type" : "cat",
        # erb 384 + df 384
        "rnn" : {"input_size": 640, "hidden_size": 256, "groups": 8, "num_layers": 2},
        "lsnr" : {"in_features": 256, "out_features": 1},
        },
    "erb_decoder": {
        "n_dec" : 4,
        "pdec1":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "pdec2":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "pdec3":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "pdec4":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "skip1" : {},
        "skip2" : {},
        "skip3" : {},
        "skip4" : {},
        "dec1": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
        "dec2": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 1], "stride": [1, 2], "padding": [0, 0], "output_padding": [0, 1]},
        "dec3": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 1], "stride": [1, 2], "padding": [0, 0], "output_padding": [0, 1]},
        "dec4": {"in_channels": 64, "out_channels": 1, "kernel_size": [1, 3], "stride": [1, 1], "padding": [0, 1]},
        # output_size = conv_ch * nb_erb//4
        "rnn" : {"input_size" : 256, "hidden_size": 256, "output_size" : 384, "num_layers" : 1 }
    },
    "df_decoder":{
        "df_bin" : 160, # nb_df 
        "df_order" : 5,
        "pconv" : {"in_channels" : 64, "out_channels": 10, "kernel_size": [1,1]}, # out_channels = 2*df_order
        "rnn" : {"input_size" : 256, "hidden_size": 256, "num_layers" : 2},
        "linear" : {"input_size": 256, "hidden_size" : 1600, "groups" : 8 }, # out_channels = 2*df_order*nb_df
        "alpha" : {"in_features": 256, "out_features": 1},
    }
}

############### Model Implementation ###############

class mpNF(nn.Module):
    def __init__(self,
                n_fft, 
                arch=arch_orig
                ):
        super(mpNF, self).__init__()

        self.n_freq = n_fft//2 + 1
        self.framewise = False
        self.arch = arch

        self.n_erb = arch["params"]["nb_erb"]
        self.n_df = arch["params"]["nb_df"]

        self.erb_encoder = ERBEncoder(arch["erb_encoder"])
        self.df_encoder = DFEncoder(arch["df_encoder"])
        self.encoder_fusion = EncoderFusion(arch["encoder_fusion"])
        self.erb_decoder = ERBDecoder(arch["erb_decoder"])
        self.df_decoder = DFDecoder(arch["df_decoder"])
        self.inter_mask = InterMask(arch["inter_mask"])

        self.n_rnn_enc = arch["encoder_fusion"]["rnn"]["num_layers"]
        self.sz_rnn_enc = arch["encoder_fusion"]["rnn"]["hidden_size"]

        self.n_rnn_erb = arch["erb_decoder"]["rnn"]["num_layers"]
        self.sz_rnn_erb = arch["erb_decoder"]["rnn"]["hidden_size"]

        self.n_rnn_df = arch["df_decoder"]["rnn"]["num_layers"]
        self.sz_rnn_df = arch["df_decoder"]["rnn"]["hidden_size"]


    def create_dummy_states(self, batch_size, device, **kwargs):
        h_enc_shape= (self.n_rnn_enc, batch_size, self.sz_rnn_enc)
        h_erb_shape = (self.n_rnn_erb, batch_size, self.sz_rnn_erb)
        h_df_shape = (self.n_rnn_df, batch_size, self.sz_rnn_df)

        h_enc = torch.zeros(*h_enc_shape).to(device)
        h_erb = torch.zeros(*h_erb_shape).to(device)
        h_df = torch.zeros(*h_df_shape).to(device)

        return h_enc, h_erb, h_df

    # TODO : Need to modified for DFN style
    def pad_x(self, x, pe_state, n_time = 2):
        if self.framewise :
            padded_x = torch.cat((pe_state, x), dim=-2)
            pe_state = padded_x[..., 1:, :]
        else:
            padded_x = F.pad(x, (0, 0, n_time-1, 0), "constant", 0.0)
        return padded_x, pe_state

    def forward(self, f_erb, f_df, h_enc = None, h_erb = None, h_df = None):
        """
        f_erb : [B,T,F_erb], erb spectrogram
        f_df: [B,T,F_df], df spectrogram
        h_erb : [B,h], erb hidden state
        h_df : [B,h], df hidden state
        h_dec : [B,h], decoder hidden state

        retrun
        m_erb : [B,T,F_erb] : ERB mask
        m_df : [B,T,F_df]   : DF mask
        h_erb : [B,h]
        h_df : [B,h]
        h_dec : [B,h]
        alpha : [B,T,1]
        """
        if h_enc is None:
            h_enc,h_erb, h_df = self.create_dummy_states(f_erb.shape[0], f_erb.device)

        #print(f"mpNF:: input {f_erb.shape} {f_df.shape} h_enc {h_enc.shape} h_erb {h_erb.shape} h_df {h_df.shape}")
        
        emb_erb, skip_erb = self.erb_encoder(f_erb)
        emb_df, skip_df = self.df_encoder(f_df)
        emb, h_enc, lsnr = self.encoder_fusion(emb_erb, emb_df, h_enc)

        m_erb, h_erb = self.erb_decoder(emb, skip_erb, h_erb)
        m_df, h_df, alpha = self.df_decoder(emb, skip_df, h_df)

        m_erb,m_df = self.inter_mask(m_erb,m_df)

        return m_erb, m_df, h_enc, h_erb, h_df, alpha
    
    # TODO
    def to_onnx(self, output_fp, device=torch.device("cpu")):
        B = 1
        T = 1
        dummy_erb = torch.randn(B, 1, T, self.n_erb).to(device)
        dummy_df = torch.randn(B, 2, T, self.n_df).to(device)
        dummy_h_enc, dummy_h_erb,dummy_h_df= self.create_dummy_states(dummy_erb.shape[0],device)

        try:
            torch.onnx.export(
                self,
                (dummy_erb,dummy_df,dummy_h_enc,dummy_h_erb,dummy_h_df),
                output_fp,
                verbose=False,
                opset_version=16,
                input_names=["ERB","DF","enc_state","erb_state","df_state"],
                output_names=["m_erb","m_df","enc_state","erb_state","df_state","alpha"],
                #dynamo=False,
            )
        except Exception as e:
            import sys
            print(f"Export failed: {e}", file=sys.stderr)
        pass

class mpNF_helper(nn.Module):
    def __init__(self, 
        sr = 16000,
        n_fft= 512, 
        n_hop= 256,
        n_erb= 24,
        n_df = 160,
        type_window = "vorbis",
        normalize=True,
        arch= arch_orig
        ):
        super(mpNF_helper, self).__init__()

        # Params
        self.n_fft = n_fft
        self.hop_length = n_hop

        self.framewise = False

        # Blocks
        self.feature = FeatureCalculator(sr,n_fft, n_hop, n_erb, n_df,type_window=type_window,normalize=normalize)
        self.model = mpNF(
            n_fft,
            arch=arch
            )
        self.masking_erb = ErbMask(
                            self.feature.erb_ifb,
                            post_filter=False,
                            beta = 0.2      
                            )
        self.masking_df = DfOp(
            n_df,
            n_order = arch["params"]["n_order"]
        )

    def forward(self, x):
        #B,L = x.shape

        # X : [B,T,F], 
        # f_erb : [B,T,F_erb],
        # f_df : [B,T,F_df]
        #print(f"mpNF_helper:: {x.shape} ",end=" ")
        X, f_erb, f_df = self.feature.analysis(x)
        #print(f"-> {X.shape} {f_erb.shape} {f_df.shape} ")

        # X : [B,T,F], 
        # f_erb : [B,1,T,F_erb],
        # f_df : [B,2,T,F_df]
        f_erb = f_erb.unsqueeze(1)
        f_df = torch.stack((f_df.real,f_df.imag),dim=1)

        # Y : [B,T,F]
        m_erb,m_df, h_enc, h_erb, h_df, alpha = self.model(f_erb, f_df, None,None,None)
        #print(f"mpNF_helper:: {m_erb.shape} {m_df.shape} {h_enc.shape} {h_erb.shape} {h_df.shape} {alpha.shape}")
        Y = self.masking_erb(X,m_erb)
        Y = self.masking_df(Y, m_df)

        # TODO
        Y[:,-1,:] = 0.0 # Last frame is not valid

        # Y : [B,L']
        y = self.feature.synthesis(Y)
        #print(f"mpNF_helper:: {Y.shape} -> {y.shape}")

        return y
    
    def forward_framewise(self,x):
        B,L = x.shape

        # X : [B,T,F], 
        # f_erb : [B,T,F_erb],
        # f_df : [B,T,F_df]
        X, f_erb, f_df = self.feature.analysis(x)

        h_enc = None
        h_erb = None
        h_df = None

        # Y : [B,T,F]
        Y = []
        for t in range(X.shape[1]) :
            i_erb = f_erb[:,t,:].unsqueeze(1)
            i_df = f_df[:,t,:].unsqueeze(1)
            m_erb,m_df, h_enc, h_erb, h_df, alpha = self.model(i_erb, i_df, h_enc, h_erb, h_df)
            Yt = self.masking_erb(X,m_erb)
            Yt = self.masking_df(X, m_df)
            Y.append(Yt)
        Y = torch.stack(Y, dim=1)
        y = self.feature.synthesis(Y)

        return y[:, :L]
    
    def to_onnx(self, output_fp, device=torch.device("cpu")):
        self.model.to_onnx(output_fp, device)

    def framewise_mode(self,flag : bool = False):
        if flag : 
            self.forward = self.forward_framewise
        self.framewise = flag
        self.model.framewise = flag
        self.masking_df.infer = flag
