#from  MTFAA.MTFAA import test
from mpSE.TRUNet import test

arch = {
    "encoder": {
        "enc1": {"in_channels": 4, "out_channels": 16, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "type_norm" : "BatchNorm2d","groups":4,"embed_dim": 260},
        "enc2": {"in_channels": 16, "out_channels": 64, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,1], "type_norm" : "BatchNorm2d", "groups": 16,"embed_dim": 512},
        "enc3": {"in_channels":64, "out_channels": 128, "kernel_size": [1, 5], "stride": [1,2], "padding": [0,2], "type_norm" : "BatchNorm2d", "groups":16,"embed_dim": 1024}

    },
    "decoder": {
        "dec3": {"in_channels": 192, "out_channels": 64, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 1], "type_norm" : "BatchNorm2d","output_padding" : (0,1)},
        "dec2": {"in_channels": 128, "out_channels": 16, "kernel_size": [1, 3], "stride": [1, 2], "padding": [0, 0], "type_norm" : "BatchNorm2d"},
        "dec1": {"in_channels": 32, "out_channels": 4, "kernel_size": [1, 5], "stride": [1, 2], "padding": [0, 2], "type_norm" : "BatchNorm2d"}

    },
    "PE": {"in_channels": 1, "out_channels" : 4, "type_norm" : "BatchNorm2d"},
    "FSA": {"in_channels": 128, "hidden_size": 8, "out_channels": 64},
    "TGRU": {"in_channels": 128, "hidden_size": 64, "out_channels": 64, "state_size": 16},
    "MEA": {"in_channels": 4, "mag_f_dim": 3}
}

test(architecture = arch,
    frame_size = 256,
    hop_size = 64,
    kernel_type = "orig"
    )
