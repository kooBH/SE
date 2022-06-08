# Torch Inference Using Multi Processor 
import torch
import argparse
import os,glob
import librosa
import soundfile as sf
import numpy as np
from utils.hparams import HParam
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method


parser = argparse.ArgumentParser()
parser.add_argument('-i','--dir_input',type=str,required=True)
parser.add_argument('-o','--dir_output',type=str,required=True)
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-d','--device',type=str,defalut='cuda:0')
parser.add_argument('-n','--num_process',type=int,defalut=8)
args = parser.parse_args()

hp = HParam(args.config)
print('NOTE::Loading configuration :: ' + args.config)

# TODO target data

# Params
device = args.device
torch.cuda.set_device(device)

num_epochs = 1
batch_size = 1

# Dirs
os.makedirs(args.dir_output,exist_ok=True)


# Model 
## TODO load model

model.share_memory()
model.eval()


def inference(batch):
    for idx in batch :

        # TODO : preprocess for data
        input = None
        with torch.no_grad():

            output = model(input)



if __name__ == '__main__':
    set_start_method('spawn')

    processes = []
    batch_for_each_process = np.array_split(range(len(list_data)),args.num_process)

    for worker in range(args.num_process):
        p = mp.Process(target=inference, args=(batch_for_each_process[worker][:],) )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()







