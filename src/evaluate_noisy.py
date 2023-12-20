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
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    global device

    ##  Dataset
    if hp.task == "SPEAR" : 
        train_dataset = DatasetSPEAR(hp,is_train=True)
        test_dataset  = DatasetSPEAR(hp,is_train=False)
    elif hp.task == "DNS":
        train_dataset = DatasetDNS(hp,is_train=True)
        test_dataset  = DatasetDNS(hp,is_train=False)
    else :
        raise Exception("ERROR::Unknown task : {}".format(hp.task))

    #### EVAL ####
    ## Metric
    metric = {}
    for m in hp.log.eval : 
        metric["{}_with_reverb".format(m)] = 0.0
        metric["{}_no_reverb".format(m)] = 0.0

    for i in tqdm(range(hp.log.n_eval)) : 
        with_reverb,no_reverb = test_dataset.get_eval(i)

        # with_reverb
        noisy_reverb = rs.load(with_reverb[0],sr=hp.data.sr)[0]
        target_reverb = rs.load(with_reverb[1],sr=hp.data.sr)[0]

        # no_reverb
        noisy_no_reverb = rs.load(no_reverb[0],sr=hp.data.sr)[0]
        target_no_reverb = rs.load(no_reverb[1],sr=hp.data.sr)[0]

        for m in hp.log.eval : 
            val_reverb = run_metric(noisy_reverb,target_reverb,m) 
            metric["{}_with_reverb".format(m)] += val_reverb
            val_no_reverb = run_metric(noisy_no_reverb,target_no_reverb,m) 
            metric["{}_no_reverb".format(m)] += val_no_reverb
        
    for m in hp.log.eval : 
        key = "{}_no_reverb".format(m)
        metric[key] /= hp.log.n_eval

        key = "{}_with_reverb".format(m)
        metric[key] /= hp.log.n_eval

    with open("log_noisy.txt",'w') as f :
        for k in metric.keys():
            f.write("'{}':'{}'\n".format(k, metric[k]))
    

