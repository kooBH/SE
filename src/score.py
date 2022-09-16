from utils.metric import PESQ

from utils.hparams import HParam
import argparse
import os
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default config")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_input','-i',type=str,required=True)
    args = parser.parse_args()


    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)
    mono = args.mono

    batch_size = 1
    num_workers = 1

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version

PESQ(estim,target)