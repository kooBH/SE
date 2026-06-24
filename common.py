import torch
import torch.nn as nn
from utils.metric import run_metric
import librosa as rs

import random
import numpy as np
import os,glob


def get_model(cfg):

    from models.model import ModelWrapper
    model = ModelWrapper(**cfg.model.params)

    return model

def run(data,model,criterion=None,device="cuda:0",ret_output=False): 
    noisy = data['noisy'].to(device)
    clean = data['clean'].to(device)
    estim = model(noisy)

    if criterion is None : 
        return estim

    loss, loss_dict = criterion(estim,clean,return_dict=True)

    if loss.isinf().any() : 
        print("Warning::There is inf in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()

    if loss.isnan().any() : 
        print("Warning::There is nan in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()
    
    if ret_output:
        return estim, loss
    else : 
        return loss, loss_dict

def evaluate(hp, model,list_data,device="cuda:0"):
    #### EVAL ####
    model.eval()
    with torch.no_grad():
        ## Metric
        metric = {}
        for m in hp.log.eval.metric : 
            metric[m] = 0.0

        for pair_data in list_data : 
            path_noisy = pair_data[0]
            path_clean = pair_data[1]
            noisy = rs.load(path_noisy,sr=hp.data.sr)[0]
            noisy = torch.unsqueeze(torch.from_numpy(noisy),0).to(device)
            estim = model(noisy).cpu().detach().numpy()[0]
            clean = rs.load(path_clean,sr=hp.data.sr)[0]

            if len(clean) > len(estim) :
                clean = clean[:len(estim)]
            else :
                estim = estim[:len(clean)]
            for m in hp.log.eval.metric : 
                val= run_metric(estim,clean,m) 
                metric["{}".format(m)] += val
            
        for m in hp.log.eval.metric : 
            key = "{}".format(m)
            metric[key] /= len(list_data)
    return metric


def set_seed(seed: int = 42):
    if seed == -1:
        #print(f"No Fixed Seed")
        return
    #print(f"Fixed Seed : {seed}")
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU 환경

    # CuDNN 설정 (완전한 determinism 보장)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 환경 변수 고정 (hash seed 등)
    os.environ["PYTHONHASHSEED"] = str(seed)

def listing_eval(root):
    dir_noisy =  os.path.join(root,"noisy")
    dir_clean =  os.path.join(root,"clean")
    list_noisy = glob.glob(os.path.join(dir_noisy,"*.wav"))
    list_eval = []
    for path_noisy in list_noisy :
        # name : noisy_1.wav
        # clean_name : clean_1.wav
        basename = os.path.basename(path_noisy)
        path_clean = os.path.join(dir_clean,basename.replace("noisy","clean"))
        list_eval.append((path_noisy,path_clean))
    return list_eval

def listing_VD(root) :
    # root      : "/home/data/Voicebank+Demand
    # dir_noisy : "/home/data/Voicebank+Demand/noisy_testset_wav"
    # dir_clean :  "/home/data/Voicebank+Demand/clean_testset_wav"
    dir_noisy = os.path.join(root,"noisy_testset_wav")
    dir_clean = os.path.join(root,"clean_testset_wav")
    list_VD = []
    for path_noisy in glob.glob(os.path.join(dir_noisy,"*.wav")) : 
        basename = os.path.basename(path_noisy)
        path_clean = os.path.join(dir_clean,basename)
        list_VD.append([path_noisy,path_clean])
    return list_VD

def listing_DNS_no_reverb(root):
    list_DNS_noisy = glob.glob(os.path.join(root,"noisy","*.wav"),recursive=True)
    list_DNS=[]
    for path_noisy in list_DNS_noisy :
        token = path_noisy.split("/")[-1]
        token = token.split("_")
        fileid = token[-1].split(".")[0]
        path_clean = os.path.join(root,"clean","clean_fileid_{}.wav".format(fileid))
        list_DNS.append((path_noisy,path_clean))

    return list_DNS