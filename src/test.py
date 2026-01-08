from common import run,get_model
from utils.hparams import HParam
import soundfile as sf
import torch

device = "cpu"

#path_default = "../config/convergence/default.yaml"
#path_config = "../config/convergence/v0.yaml"

#hp = HParam(path_config,path_default,merge_except=["architecture"])
#hp = HParam(path_config,path_default)
#model = get_model(hp,device=device)

"""
from mpSE.mpNC_S import mpNC_S_wrapper
model = mpNC_S_wrapper(frame_size=512, hop_size=128)
x = torch.rand(2,49152)

#y,h = model(x)
y = model(x)
print("{} -> {}".format(x.shape,y.shape))
"""

from mpNF.mpNF import mpNF_helper
import time

def test_mpNF() : 
    arch = {
    "params":{
        "nb_erb" : 24,
        "nb_df" : 257,
        "n_causal" : 2,
        "n_order" : 5
    },
    "erb_encoder": {
        "n_causal" : 2,
        "n_enc" : 3,
        "type_encoder" : "ConvV1",
        # First kernel (1 + n_causal,3)
        "enc1": {"in_channels": 1, "out_channels": 64,"hidden_size" : 24 , "kernel_size": [3, 3], "stride": [1,1], "padding": [0,1],"groups" : -1,"bias" : False},
        "enc2": {"in_channels": 64, "out_channels": 64, "hidden_size" : 12,"kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": -1,"bias" :False },
        "enc3": {"in_channels": 64, "out_channels": 64,"hidden_size" : 6, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups":-1, "bias" : False},
    },
    "df_encoder": {
        "n_causal" : 2,
        "n_enc" : 4,
        "type_encoder" : "ConvV1",
        # First kernel (1 + n_causal,3)
        "enc1": {"in_channels": 2, "out_channels": 64,"hidden_size" : 257, "kernel_size": [3, 3], "stride": [1,1], "padding": [0,1], "groups": -1, "bias" : False},
        "enc2": {"in_channels": 64, "out_channels": 64,"hidden_size" : 129 , "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": -1, "bias" : False},
        "enc3": {"in_channels": 64, "out_channels": 64,"hidden_size" : 65, "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": -1, "bias" : False},
        "enc4": {"in_channels": 64, "out_channels": 64,"hidden_size" : 33 , "kernel_size": [1, 3], "stride": [1,2], "padding": [0,1], "groups": -1, "bias" : False}
    },
    "encoder_fusion": {
        "type_emb" : "FSA",
        #"linear_df" : {"input_size" : 4160, "hidden_size" : 320, "groups" : 8},

        "type_fusion" : "FSA",
        # df : 64 * 33 = 2112
        # erb : 64 * 6 = 384
        "FSA" : {"erb_size": 64, "df_size": 64, "erb_f" : 6, "df_f":33,"hidden_size": 256, "num_heads": 4},

        "rnn" : {"input_size": 512, "hidden_size": 512, "groups": 8, "num_layers": 2},
        "lsnr" : {"in_features": 512, "out_features": 1},
        },
    "erb_decoder": {
        "type_fblock" : "None",
        "FSA" : {"input_size": 512, "hidden_size": 512, "num_heads": 4, "type_PE": "PositionalEncoding", "type_activation": "ReLU"},
        # output_size = conv_ch * nb_erb//4
        "rnn" : {"input_size" : 512, "hidden_size": 512, "output_size" : 384, "num_layers" : 1 },
        "n_dec" : 3,
        "pdec1":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "pdec2":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "pdec3":{"in_channels" : 64, "out_channels": 64, "kernel_size": [1, 1]},
        "skip1" : {},
        "skip2" : {},
        "skip3" : {},
        "dec1": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 1],"output_padding": [0, 1]},
        "dec2": {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 1], "stride": [1, 2], "padding": [0, 0], "output_padding": [0, 1]},
        "dec3": {"in_channels": 64, "out_channels": 1, "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0]}
    },
    "df_decoder":{
        "df_bin" : 257, # nb_df 
        "df_order" : 5,
        "pconv" : {"in_channels" : 64, "out_channels": 10, "kernel_size": [1,1]}, # out_channels = 2*df_order
        "rnn" : {"input_size" : 512, "hidden_size": 256, "num_layers" : 2},
        "linear" : {"input_size": 256, "hidden_size" : 2570, "groups" : 1 }, # out_channels = 2*df_order*nb_df
        "alpha" : {"in_features": 256, "out_features": 1},
    },
      "inter_mask" : {
      "type_cross" : "Linear",
      "Linear" : {"erb_size" : 24,"df_size":2570, "type_activation" : "None", "bias":False}
    }
}

    print(" == mpNF ==")
    model = mpNF_helper(
        n_erb=24,
        n_df = 257,
        type_window = "sine",
        arch=arch).to(device)
    sample_rate = 16000 
    duration = 1.0  
    n_sample = 4096*12
    freq = 440.0 

    #t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
    t = torch.linspace(0, 1, n_sample, dtype=torch.float32)
    x = torch.sin(2 * torch.pi * freq * t)*0.3
    x += torch.sin(4 * torch.pi * freq * t - 0.3*torch.pi) * 0.15
    x += torch.sin(8 * torch.pi * freq * t - 0.3*torch.pi) * 0.07
    #x = torch.rand(n_sample)-0.5
    x = x.unsqueeze(0)
    x= x.repeat(64, 1)

    print(f"{x.shape}")
    output = model(x)
    print(output.shape)  # Output shape

    #with torch.no_grad():
    #    sf.write("input.wav", x[0].numpy(), sample_rate)
    #    sf.write("output.wav", output[0].numpy(), sample_rate)

    """
    print(" == Framewise Mode ==")
    model.framewise_mode(True)
    x = torch.randn(1, 16000)  # Example input
    output = model(x)
    print(output.shape)  # Output shape
    """

def test_lsnr() :
    from LSNR.LSNRNet import LSNRNetHelper

    default_arch = {
        "spectrum": {
            "type" : "magnitude",
            "LPC" : True
        },
        "encoder" : {
            "enc1" : {
                "in_channels": 1, 
                "out_channels": 64,
                "kernel_size": [1, 5],
                "stride": [1,2],
                "padding": [0,0]
            },
            "enc2" : {
                "in_channels": 64, 
                "out_channels": 128,
                "kernel_size": [1, 5],
                "stride": [1,2],
                "padding": [0,0]
            },
            "enc3" : {
                "in_channels": 128, 
                "out_channels": 256,
                "kernel_size": [1, 5],
                "stride": [1,2],
                "padding": [0,0]
            }
        },
        "decoder" : {
            "dec1" : {
                "in_channels": 512, 
                "out_channels": 128,
                "kernel_size": [1, 5],
                "stride": [1,2],
                "padding": [0,0],
                "output_padding" :[0,1]
            },
            "dec2" : {
                "in_channels": 256, 
                "out_channels": 64,
                "kernel_size": [1,5],
                "stride": [1,2],
                "padding": [0,0],
            },
            "dec3" : {
                "in_channels": 128, 
                "out_channels": 1,
                "kernel_size": [1, 5],
                "stride": [1,2],
                "padding": [0,0],
                "output_padding" :[0,0]
            }
        },
        "temporal" : {
            "hidden_size" : 256
        },
        "frequential" : {
            "hidden_size" : 256
        },
        "mask":{
            "type" : "SNRAwareMagMaskingV2",
            "hidden_size" : 257
        },
        "LSNR" : {
            "type" : "lsnr_estimator4",
            "location" : "bottleneck3", # bottleneck1, bottleneck2, bottleneck3
            "input_channels" : 256,
            "hidden_size" : 29 # 29 * 256
        }
    }

    x = torch.randn(1, 16000)  # Example input
    m = LSNRNetHelper(architecture=default_arch)

    y,lsrn = m(x)

#from mpNF.mpTF import test_mpTF
from mpSE.SE_T1 import test_SE_T1

if __name__ == "__main__":
    #test_mpNF()
    #test_lsnr()
    #test_mpTF()
    #test_SE_T1()

    from mpSE.mpNC import test_mpNC
    test_mpNC()