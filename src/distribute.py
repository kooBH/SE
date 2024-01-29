import os, argparse
import glob
from utils.hparams import HParam
import numpy as np
import torch
import torch.onnx
from common import get_model
from utils.metric import run_metric
import librosa as rs

dir_voice_demand = ""
dir_dns2020      = ""

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                    help="default configuration")
    parser.add_argument('--version',"-v", type=str, required=True)
    parser.add_argument('--task',"-t", type=str, required=True)
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    args = parser.parse_args()

    hp = HParam(args.config,args.default,merge_except=["architecture"])
    print("NOTE::Loading configuration : "+args.config)
    task = args.task

    model = get_model(hp,"cpu")
    model.load_state_dict(torch.load(args.chkpt, map_location="cpu"))
    model.eval()

    ## Prep Test Dataset


    # TODO
    list_VD = glob.glob(os.path.join(dir_voice_demand,"**","*.wav",recursive=True))
    list_DNS = glob.glob(os.path.join(dir_dns2020,"**","*.wav",recursive=True))


    ### ONNX

    n_fft = hp.audio.n_fft
    n_hop = hp.audio.n_hop

    os.makedirs("./chkpt",exist_ok=True)

    # torch to ONNX
    version = args.version

    name = task + "_" + version

    ## tracing
    n_feat = 2

    input = torch.rand(1,n_fft//2+1,1,n_feat)
    """
    traced_model = torch.jit.trace(model, input)
    traced_model.save('./chkpt/rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
    """

    print("ONXX Export")
    model.to_onnx("./chkpt/"+name+".onnx")

    # Eval
    for idx,pair_data in enumerate(list_VD) : 
        path_noisy = pair_data[0]
        path_clean = pair_data[1]
        noisy = rs.load(path_noisy,sr=hp.data.sr)[0]
        noisy = torch.unsqueeze(torch.from_numpy(noisy),0).to(device)
        estim = model(noisy).cpu().detach().numpy()[0]
        clean = rs.load(path_clean,sr=hp.data.sr)[0]



    ### N_PARAM
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters : {:,}".format(n_parameters))



    # Flops
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model.helper, input)
    print("--total--")
    total_flops = flops.total()
    print('{:,}'.format(total_flops))



