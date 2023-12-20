import sys, time, os, argparse
from utils.hparams import HParam
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
from common import get_model

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                    help="default configuration")
    parser.add_argument('--version',"-v", type=str, required=True)
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    args = parser.parse_args()

    hp = HParam(args.config,args.default,merge_except=["architecture"])
    print("NOTE::Loading configuration : "+args.config)

    model = get_model(hp,"cpu")
    model.load_state_dict(torch.load(args.chkpt, map_location="cpu"))
    model.eval()

    n_fft = hp.audio.n_fft
    n_hop = hp.audio.n_hop

    os.makedirs("./chkpt",exist_ok=True)

    # torch to ONNX
    version = args.version

    name = hp.model.type + "_" + version

    ## tracing
    n_feat = 2
    if hp.model.use_mag :
        n_feat = 3

    input = torch.rand(1,n_fft//2+1,1,n_feat)
    """
    traced_model = torch.jit.trace(model, input)
    traced_model.save('./chkpt/rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
    """

    print("ONXX Export")
    model.to_onnx("./chkpt/"+name+".onnx")

    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model.helper, input)
    print("--total--")
    total_flops = flops.total()
    print('{:,}'.format(total_flops))
    #print("--by operator--")
    #print(flops.by_operator())
    #print("--by module--")
    #print(flops.by_module())
    #print("--by module and operator--")
    #print(flops.by_module_and_operator())
