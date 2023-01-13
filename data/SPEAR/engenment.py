import librosa as rs
import soundfile as sf
import numpy as np

from glob import glob
from os.path import join
import os

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

sr = 16000
len_sec = 5
overlap = 0.25

ratio_shift= 1-overlap

len_seg = int(len_sec*sr)

def process(idx) : 
    path_noisy = list_noisy[idx]

    # Dataset_2
    dir_dataset = path_noisy.split("/")[-4]
    # Session_1
    dir_session = path_noisy.split("/")[-2]
    # 00
    id_utt = path_noisy.split("_")[-1][1:3]

    # ex : /home/data/kbh/SPEAR_TARGET/Train/Dataset_2/Reference_Audio/Session_1/00/ref_D2_S1_M00_ID4.wav
    # ex : /home/data/kbh/SPEAR_TARGET/Train/Dataset_2/Reference_Audio/Session_1/00/ref_D2_S1_M00_ID6.wav
    
    list_clean = glob(join(root_target,train_dev,dir_dataset,"Reference_Audio",dir_session,id_utt,"*.wav"))

    ## Load data
    idx_noisy = np.random.randint(6)
    idx_clean = np.random.randint(2)

    noisy = rs.load(path_noisy,sr=sr,mono=False,res_type="fft")[0]
    clean = None

    for path_clean in list_clean : 
        if clean is None  : 
            clean = rs.load(path_clean,sr=sr,mono=False,res_type="fft")[0]
        else :
            clean += rs.load(path_clean,sr=sr,mono=False,res_type="fft")[0]

    # EnSeg & Save
    idx = 0
    while idx < noisy.shape[1] : 
        for i in range(noisy.shape[0]) :
            sf.write(join(root_out,train_dev,"noisy","{}_{}_{}_ch_{}_seg_{}.wav".format(dir_dataset,dir_session,id_utt,i,idx)),noisy[i,idx:idx+len_seg],sr)

        idx += int(len_seg * ratio_shift)

    idx = 0
    while idx < clean.shape[1] : 
        for i in range(clean.shape[0]) :
            sf.write(join(root_out,train_dev,"clean","{}_{}_{}_ch_{}_seg_{}.wav".format(dir_dataset,dir_session,id_utt,i,idx)),clean[i,idx:idx+len_seg],sr)

        idx += int(len_seg * ratio_shift)



if __name__ == "__main__" : 
    cpu_num = int(cpu_count()/2)

    root_noisy  = "/home/data2/kbh/SPEAR_INPUT"
    root_target = "/home/data2/kbh/SPEAR_TARGET"

    root_out = "/home/data2/kbh/SPEAR_seg_5sec"

    os.makedirs(root_out,exist_ok=True)
    os.makedirs(root_out+"/Train/noisy",exist_ok=True)
    os.makedirs(root_out+"/Train/clean",exist_ok=True)
    os.makedirs(root_out+"/Dev/noisy",exist_ok=True)
    os.makedirs(root_out+"/Dev/clean",exist_ok=True)

    train_dev = "Train"
    list_noisy = []
    list_noisy += glob(join(root_noisy,train_dev,"Dataset_2","Microphone_Array_Audio","*","*.wav"))
    list_noisy += glob(join(root_noisy,train_dev,"Dataset_3","Microphone_Array_Audio","*","*.wav"))
    list_noisy += glob(join(root_noisy,train_dev,"Dataset_4","Microphone_Array_Audio","*","*.wav"))

    arr = list(range(len(list_noisy)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='Train'))


    train_dev = "Dev"
    list_noisy = []
    list_noisy += glob(join(root_noisy,train_dev,"Dataset_2","Microphone_Array_Audio","*","*.wav"))
    list_noisy += glob(join(root_noisy,train_dev,"Dataset_3","Microphone_Array_Audio","*","*.wav"))
    list_noisy += glob(join(root_noisy,train_dev,"Dataset_4","Microphone_Array_Audio","*","*.wav"))

    arr = list(range(len(list_noisy)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='Dev'))