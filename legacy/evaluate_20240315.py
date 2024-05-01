import torch
import argparse
import os
import numpy as np
import librosa as rs

from Dataset.DatasetGender import DatasetGender
from Dataset.DatasetSPEAR import DatasetSPEAR
from Dataset.DatasetDNS import DatasetDNS

from utils.hparams import HParam
from utils.metric import run_metric

from tqdm.auto import tqdm

from common import run,get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration {} based on {}".format(args.config,args.default))
    global device

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    ##  Dataset
    if hp.task == "SPEAR" : 
        train_dataset = DatasetSPEAR(hp,is_train=True)
        test_dataset  = DatasetSPEAR(hp,is_train=False)
    elif hp.task == "DNS":
        train_dataset = DatasetDNS(hp,is_train=True)
        test_dataset  = DatasetDNS(hp,is_train=False)
    else :
        raise Exception("ERROR::Unknown task : {}".format(hp.task))

    model = get_model(hp,device=device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        try : 
            model.load_state_dict(torch.load(args.chkpt, map_location=device)["model"])
        except KeyError :
            model.load_state_dict(torch.load(args.chkpt, map_location=device))
    cnt_log = 0

    
    #### EVAL ####
    model.eval()
    with torch.no_grad():
        ## Metric
        metric = {}
        for m in hp.log.eval : 
            metric["{}_with_reverb".format(m)] = 0.0
            metric["{}_no_reverb".format(m)] = 0.0

        for i in tqdm(range(hp.log.n_eval)) : 
            with_reverb,no_reverb = test_dataset.get_eval(i)

            # with_reverb
            noisy_reverb = rs.load(with_reverb[0],sr=hp.data.sr)[0]
            noisy_reverb = torch.unsqueeze(torch.from_numpy(noisy_reverb),0).to(device)
            estim_reverb = model(noisy_reverb).cpu().detach().numpy()

            target_reverb = rs.load(with_reverb[1],sr=hp.data.sr)[0]

            # no_reverb
            noisy_no_reverb = rs.load(no_reverb[0],sr=hp.data.sr)[0]
            noisy_no_reverb = torch.unsqueeze(torch.from_numpy(noisy_no_reverb),0).to(device)
            estim_no_reverb = model(noisy_no_reverb).cpu().detach().numpy()

            target_no_reverb = rs.load(no_reverb[1],sr=hp.data.sr)[0]

            for m in hp.log.eval : 
                val_reverb = run_metric(estim_reverb[0],target_reverb,m) 
                metric["{}_with_reverb".format(m)] += val_reverb
                val_no_reverb = run_metric(estim_no_reverb[0],target_no_reverb,m) 
                metric["{}_no_reverb".format(m)] += val_no_reverb
            
        for m in hp.log.eval : 
            key = "{}_no_reverb".format(m)
            metric[key] /= hp.log.n_eval

            key = "{}_with_reverb".format(m)
            metric[key] /= hp.log.n_eval

    with open("log_{}.txt".format(version),'w') as f :
        for k in metric.keys():
            f.write("'{}':'{}'\n".format(k, metric[k]))
    

