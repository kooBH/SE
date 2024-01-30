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
dir_dns2020      = "/home/data/kbh/DNS2020/ICASSP_dev_test_set/track_1/synthetic"

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
        # e.g.
        # synthetic_emotion_1792_snr19_tl-35_fileid_19.wav
        # -> synthetic_emotion_clean_filed_19.wav
        #  
        # synthetic_french_ancientgreek_epigram68-03_train_26209_5_snr16_tl-25_fileid_64.wav
        # -> synthtic_clean_fileid_64.wav
        token = path_noisy.split("/")[-1]
        token = token.split("_")
        fileid = token[-1].split(".")[0]
        if token[1] == "emotion" :
            path_clean = os.path.join(dir_dns2020,"clean","synthetic_emotion_clean_fileid_{}.wav".format(fileid))
        elif token[1] == "singing" : 
            path_clean = os.path.join(dir_dns2020,"clean","synthetic_singing_clean_fileid_{}.wav".format(fileid))
        else :
            path_clean = os.path.join(dir_dns2020,"clean","synthetic_clean_fileid_{}.wav".format(fileid))
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

    input = torch.rand(1,n_fft//2+1,1,n_feat)
    """
    traced_model = torch.jit.trace(model, input)
    traced_model.save('./chkpt/rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
    """

    print("ONXX Export")
    model.to_onnx("./chkpt/"+name+".onnx")

    # Eval for DNS2020 dev synthetic no reverb
    print("Eval DNS2020 dev : {}".format(len(list_DNS)))
    hp.log.eval = ["PESQ","SISDR","STOI","SigMOS"]
    metric_DNS = evaluate(hp,model,list_DNS,"cpu")


    # Eval for Voice+Demand
    print("Eval Voice+Demand : {}".format(len(list_VD)))
    hp.log.eval = ["PESQ","SISDR","STOI"]
    metric_VD = evaluate(hp,model,list_VD,"cpu")

    ### N_PARAM
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters : {:,}".format(n_parameters))

    # Flops
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model.helper, input)
    print("--total--")
    total_flops = flops.total()
    print('{:,}'.format(total_flops))

    with open("./log/"+name+".txt","w") as f :
        f.write("VD : \n")
        for k,v in metric_VD.items() :
            f.write("{} : {}\n".format(k,v))
        f.write("DNS : \n")
        for k,v in metric_DNS.items() :
            f.write("{} : {}\n".format(k,v))
        f.write("N_PARAM : {:,}\n".format(n_parameters))
        f.write("FLOPS : {:,}\n".format(total_flops))



