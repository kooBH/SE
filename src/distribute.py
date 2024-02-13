import os, argparse
import glob
from utils.hparams import HParam
import numpy as np
import torch
import torch.onnx
from common import get_model, evaluate
from utils.metric import run_metric
import librosa as rs

dir_voice_demand = "/home/data/kbh/Voicebank+Demand"
dir_dns2020      = "/home/data/kbh/DNS2020/test_set/synthetic/no_reverb"

flag_get_score = True

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
    list_VD_clean = glob.glob(os.path.join(dir_voice_demand,"clean_testset_wav","*.wav"),recursive=True) 

    list_VD=[]
    for path_clean in list_VD_clean :
        path_noisy = path_clean.replace("clean_testset_wav","noisy_testset_wav")
        list_VD.append((path_noisy,path_clean))
    
    list_DNS_noisy = glob.glob(os.path.join(dir_dns2020,"noisy","*.wav"),recursive=True)

    list_DNS=[]
    for path_noisy in list_DNS_noisy :
        token = path_noisy.split("/")[-1]
        token = token.split("_")
        fileid = token[-1].split(".")[0]
        path_clean = os.path.join(dir_dns2020,"clean","clean_fileid_{}.wav".format(fileid))
        list_DNS.append((path_noisy,path_clean))

    ### ONNX
    n_fft = hp.audio.n_fft
    n_hop = hp.audio.n_hop

    os.makedirs("./chkpt",exist_ok=True)
    os.makedirs("./log",exist_ok=True)

    # torch to ONNX
    version = args.version

    name = task + "_" + version

    ## tracing
    n_feat = 2 # complex(real,imag)
    T = 125

    #input = torch.rand(1,n_fft//2+1,1,n_feat)
    input = torch.rand(1,n_fft//2+1,T,n_feat)
    """
    traced_model = torch.jit.trace(model, input)
    traced_model.save('./chkpt/rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
    """

    print("ONXX Export")
    model.to_onnx("./chkpt/"+name+".onnx")

    if flag_get_score : 
        # Eval for DNS2020 dev synthetic no reverb
        print("Eval DNS2020 dev : {}".format(len(list_DNS)))
        hp.log.eval = ["PESQ","SISDR","SigMOS","STOI", "PESQ_WB","PESQ_NB"]
        metric_DNS = evaluate(hp,model,list_DNS,"cpu")


        # Eval for Voice+Demand
        print("Eval Voice+Demand : {}".format(len(list_VD)))
        hp.log.eval = ["PESQ","SISDR","STOI","PESQ_WB","PESQ_NB"]
        metric_VD = evaluate(hp,model,list_VD,"cpu")

    ### N_PARAM
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters : {:,}".format(n_parameters))

    # Flops
    #from fvcore.nn import FlopCountAnalysis
    #flops = FlopCountAnalysis(model.helper, input)
    #total_flops = flops.total()


    model.helper.eval()
    # https://github.com/Lyken17/pytorch-OpCounter
    #from thop import profile
    #input = torch.rand(1,n_fft//2+1,T,n_feat)
    #macs_thop, params_thop = profile(model.helper, inputs=(input,))
    #print("thop : MACS : {} | Param : {}".format(macs_thop,params_thop))

    # https://github.com/sovrasov/flops-counter.pytorch
    from ptflops import get_model_complexity_info
    macs_ptflos, params_ptflops = get_model_complexity_info(model.helper, (n_fft//2+1,T,n_feat), as_strings=False,                                           print_per_layer_stat=True, verbose=True)   
    print("ptflops : MACS {} |  PARAM {}".format(macs_ptflos,params_ptflops))


    if flag_get_score : 
        with open("./log/"+name+".txt","w") as f :
            f.write("VD : \n")
            for k,v in metric_VD.items() :
                f.write("{} : {}\n".format(k,v))
            f.write("DNS : \n")
            for k,v in metric_DNS.items() :
                f.write("{} : {}\n".format(k,v))
            f.write("N_PARAM : {}\n".format(n_parameters))
            f.write("MACS : {}\n".format(macs_ptflos))
            f.write("PARAM_ptflops : {}\n".format(params_ptflops))
        #    f.write("Flops : {}\n".format(total_flops))



