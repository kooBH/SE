import os, argparse
import glob
from utils.hparams import HParam
import numpy as np
import torch
import torch.onnx
from common import get_model, evaluate
from utils.metric import run_metric
import librosa as rs
import soundfile as sf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

"""
dir_voice_demand = "/home/data/kbh/Voicebank+Demand"
dir_dns2020      = "/home/data/kbh/DNS2020/test_set/synthetic/no_reverb"
dir_dns2020_reverb      = "/home/data/kbh/DNS2020/test_set/synthetic/with_reverb"

dir_LibriSpeech_SNR10 = "/home/data/kbh/LibriSpeech_noisy/SNR10"
dir_LibriSpeech_SNR5 = "/home/data/kbh/LibriSpeech_noisy/SNR5"
dir_LibriSpeech_SNR0 = "/home/data/kbh/LibriSpeech_noisy/SNR0"
"""

DIR_DB = "/home/data/"

dir_voice_demand = DIR_DB+"/Voicebank+Demand"
dir_dns2020      = DIR_DB+"/DNS2020/test_set/synthetic/no_reverb"
dir_dns2020_reverb      = DIR_DB+"DNS2020/test_set/synthetic/with_reverb"

dir_LibriSpeech_SNR10 = DIR_DB+"/LibriSpeech_noisy/SNR10"
dir_LibriSpeech_SNR5 = DIR_DB+"/LibriSpeech_noisy/SNR5"
dir_LibriSpeech_SNR0 = DIR_DB+"/LibriSpeech_noisy/SNR0"

list_sample = ["sample/HINT_SNR0_mono_16kHz.wav", "sample/TV_16kHz_mono.wav","sample/MH_mpB_PF_upscale4.wav"]

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                    help="default configuration")
    parser.add_argument('--version',"-v", type=str, required=True)
    parser.add_argument('--task',"-t", type=str, required=True)
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument("--eval", action="store_true", help="Enable debug mode")
    parser.add_argument('--mode', type=str,default="score")
    args = parser.parse_args()

    flag_get_score = args.eval
    mode = args.mode

    hp = HParam(args.config,args.default,merge_except=["architecture"])
    print("NOTE::Loading configuration : "+args.config)
    task = args.task

    model = get_model(hp,"cpu")

    model.load_state_dict(torch.load(args.chkpt, map_location="cpu"))
    model.eval()

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
    if n_hop == 128 : 
        T = 125
    elif n_hop == 64 :
        T = 250
    elif n_hop == 256 :
        T = 63
    else :
        raise ValueError("n_hop not supported : {}".format(n_hop))

    #input = torch.rand(1,n_fft//2+1,1,n_feat)
    """
    traced_model = torch.jit.trace(model, input)
    traced_model.save('./chkpt/rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
    """

#        print("WER LibriSpeech SNR10 : {}".format(len(list_LibriSpeech_SNR10)))

#        print("WER LibriSpeech SNR5 : {}".format(len(list_LibriSpeech_SNR5)))

#        print("WER LibriSpeech SNR0  : {}".format(len(list_LibriSpeech_SNR0)))

    ### N_PARAM
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters : {:,}".format(n_parameters))

    # Flops
    #from fvcore.nn import FlopCountAnalysis
    #flops = FlopCountAnalysis(model.helper, input)
    #total_flops = flops.total()

    
    input_shape = (16000,)
    input = torch.randn(input_shape)
    # https://github.com/Lyken17/pytorch-OpCounter
    #from thop import profile
    #input = torch.rand(1,n_fft//2+1,T,n_feat)
    #macs_thop, params_thop = profile(model.helper, inputs=(input,))
    #print("thop : MACS : {} | Param : {}".format(macs_thop,params_thop))

    # https://github.com/sovrasov/flops-counter.pytorch
    from ptflops import get_model_complexity_info
    macs_ptflos, params_ptflops = get_model_complexity_info(model, input_shape, as_strings=False,print_per_layer_stat=False, verbose=False) 

    print("ptflops : MACS {} |  PARAM {}".format(macs_ptflos,params_ptflops))



    if flag_get_score : 
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

        list_DNS_reverb_noisy = glob.glob(os.path.join(dir_dns2020_reverb,"noisy","*.wav"),recursive=True)
        list_DNS_reverb=[]
        for path_noisy in list_DNS_reverb_noisy :
            token = path_noisy.split("/")[-1]
            token = token.split("_")
            fileid = token[-1].split(".")[0]
            path_clean = os.path.join(dir_dns2020_reverb,"clean","clean_fileid_{}.wav".format(fileid))
            list_DNS_reverb.append((path_noisy,path_clean))

        list_LibriSpeech_SNR10 = glob.glob(os.path.join(dir_LibriSpeech_SNR10,"*.wav"),recursive=True)

        list_LibriSpeech_SNR5 = glob.glob(os.path.join(dir_LibriSpeech_SNR5,"*.wav"),recursive=True)

        list_LibriSpeech_SNR0 = glob.glob(os.path.join(dir_LibriSpeech_SNR0,"*.wav"),recursive=True)

        hp.log.eval = ["PESQ","SISDR","STOI","PESQ_WB","PESQ_NB","DNSMOS","UTMOS"]
        # Eval for DNS2020 dev synthetic no reverb
        print("Eval DNS2020 dev : {}".format(len(list_DNS)))
        metric_DNS = evaluate(hp,model,list_DNS,"cpu")

        print("Eval DNS2020 reverb dev : {}".format(len(list_DNS_reverb)))
        metric_DNS_reverb = evaluate(hp,model,list_DNS_reverb,"cpu")

        # Eval for Voice+Demand
        print("Eval Voice+Demand : {}".format(len(list_VD)))
        metric_VD = evaluate(hp,model,list_VD,"cpu")


        # Eval for Samples
        dir_sample_out = "/home/data/kbh/SE_sample_out"
        os.makedirs(dir_sample_out,exist_ok=True)
        #plt.figure()
        #fig,ax = plt.subplots(len(list_sample),2)
        DNSMOS=[]
        UTMOS=[]
        for path in list_sample : 
            print("Processing Sample : {}".format(path))
            x = rs.load(path,sr=16000)[0]
            x_tensor = torch.tensor(x).unsqueeze(0).to("cpu")
            with torch.no_grad() :
                y= model(x_tensor)
                y = y.squeeze(0).cpu().detach().numpy()

            base_name = os.path.basename(path).replace(".wav","")
            base_name = f"{base_name}_{version}.wav"

            DNSMOS.append((base_name,run_metric(y, None, "DNSMOS",fs=16000)))
            UTMOS.append((base_name,run_metric(y, None, "UTMOS",fs=16000)))

            sf.write(os.path.join(dir_sample_out,base_name),y,16000)


        with open("./log/"+name+".txt","w") as f :
            f.write("VD : \n")
            for k,v in metric_VD.items() :
                f.write("{} : {}\n".format(k,v))
            f.write("DNS no_reverb: \n")
            for k,v in metric_DNS.items() :
                f.write("{} : {}\n".format(k,v))
            f.write("DNS with_reverb : \n")
            for k,v in metric_DNS_reverb.items() :
                f.write("{} : {}\n".format(k,v))
            f.write("N_PARAM : {}\n".format(n_parameters))
            f.write("MACS : {}\n".format(macs_ptflos))
            f.write("PARAM_ptflops : {}\n".format(params_ptflops))
            #f.write("Flops : {}\n".format(total_flops))
            #f.write("{},{},{},{},{},{},{},{}".format(metric_VD["PESQ_WB"],metric_VD["SISDR"],metric_VD["STOI"],metric_DNS["PESQ_WB"],metric_DNS["SISDR"],metric_DNS["STOI"],metric_DNS_reverb["PESQ_WB"],metric_DNS_reverb["SISDR"],metric_DNS_reverb["STOI"],n_parameters,macs_ptflos))

            # samples
            for sample_name, dnsmos in DNSMOS :
                f.write("DNSMOS {} : {}\n".format(sample_name, dnsmos))
            for sample_name, utmos in UTMOS :
                f.write("UTMOS {} : {}\n".format(sample_name, utmos))

            # Copy purpose
            if mode == "score" :
                f.write(f"{metric_VD['PESQ_WB']}\t{metric_VD['SISDR']}\t{metric_VD['STOI']}\t{metric_VD['DNSMOS']}\t{metric_DNS['PESQ_WB']}\t{metric_DNS['SISDR']}\t{metric_DNS['STOI']}\t{metric_DNS['DNSMOS']}\t")
                f.write(f"{metric_DNS_reverb['PESQ_WB']}\t{metric_DNS_reverb['SISDR']}\t{metric_DNS_reverb['STOI']}\t{metric_DNS_reverb['DNSMOS']}\t")
                f.write(f"{n_parameters}\t{macs_ptflos}\t")
                f.write("\n")



            elif mode == "real" : 
                f.write(f"{metric_VD['PESQ_WB']}\t{metric_VD['SISDR']}\t{metric_VD['STOI']}\t{metric_VD['DNSMOS']}\t{metric_DNS['PESQ_WB']}\t{metric_DNS['SISDR']}\t{metric_DNS['STOI']}\t{metric_DNS['DNSMOS']}\t")
                f.write(f"{metric_DNS_reverb['PESQ_WB']}\t{metric_DNS_reverb['SISDR']}\t{metric_DNS_reverb['STOI']}\t{metric_DNS_reverb['DNSMOS']}\t")
                f.write(f"{DNSMOS[0][1]}\t{UTMOS[0][1]}\t")
                f.write(f"{DNSMOS[1][1]}\t{UTMOS[1][1]}\t")
                f.write(f"{DNSMOS[2][1]}\t{UTMOS[2][1]}\t")
                f.write(f"{n_parameters}\t{macs_ptflos}\t")
                f.write("\n")

    else : 
        with open("./log/"+name+".txt","w") as f :
            f.write("N_PARAM : {}\n".format(n_parameters))
            f.write("MACS : {}\n".format(macs_ptflos))
            f.write("PARAM_ptflops : {}\n".format(params_ptflops))

    print("ONXX Export")
    model.to_onnx("./chkpt/"+name+".onnx")







