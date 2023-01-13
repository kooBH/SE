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
    # ex : /home/data/kbh/SPEAR_INPUT/Dev/Dataset_2/DOA_sources/Session_10/doa_D2_S10_M00_ID4.csv
    # ex : /home/data/kbh/SPEAR_INPUT/Dev/Dataset_2/DOA_sources/Session_10/doa_D2_S10_M00_ID6.csv
    path_doa = list_doa[idx]

    name_doa = path_doa.split("/")[-1]

    # Dataset_2
    dir_dataset = path_doa.split("/")[-4]
    # Session_1
    dir_session = path_doa.split("/")[-2]
    # 00
    id_utt = path_doa.split("_")[-2][1:3]
    # ID6.csv -> 6
    id_spk = path_doa.split("_")[-1][2]

    # doa,D2,S10,M00,ID6.csv
    name_seg = name_doa.split("_")
    D = name_seg[1]
    S = name_seg[2]
    M = name_seg[3]

    # /home/data/kbh/SPEAR_INPUT/Dev/Dataset_1/Microphone_Array_Audio/Session_10/array_D1_S10_M00.wav
    path_noisy = join(root_noisy,train_dev,dir_dataset,"Microphone_Array_Audio",dir_session,"array_{}_{}_{}.wav".format(D,S,M))
    
    # ex /home/data/kbh/SPEAR_TARGET/Dev/Dataset_1/Reference_Audio/Session_10/00/ref_D1_S10_M00_ID4.wav
    path_clean = join(root_target,train_dev,dir_dataset,"Reference_Audio",dir_session,id_utt,"ref_{}_{}_{}_ID{}.wav".format(D,S,M,id_spk))

    # ex : /home/data/kbh/SPEAR_CDR/Dev/Dataset_1/Session_10/CDR_D1_S10_M00_ID4.wav
    path_cdr =  join(root_cdr,train_dev,dir_dataset,dir_session,"CDR_{}_{}_{}_ID{}.wav".format(D,S,M,id_spk))

    ## Load data
    noisy = rs.load(path_noisy,sr=sr,mono=True,res_type="fft")[0]
    clean = rs.load(path_clean,sr=sr,mono=True,res_type="fft")[0]
    cdr = rs.load(path_cdr,sr=sr,mono=True,res_type="fft")[0]

    # EnSeg & Save
    idx = 0
    while idx < noisy.shape[0] : 
        sf.write(join(root_out,train_dev,"noisy","{}_{}_{}_{}_seg_{}.wav".format(dir_dataset,dir_session,id_utt,id_spk,idx)),noisy[idx:idx+len_seg],sr)
        idx += int(len_seg * ratio_shift)

    idx = 0
    while idx < clean.shape[0] : 
        sf.write(join(root_out,train_dev,"clean","{}_{}_{}_{}_seg_{}.wav".format(dir_dataset,dir_session,id_utt,id_spk,idx)),clean[idx:idx+len_seg],sr)
        idx += int(len_seg * ratio_shift)

    idx = 0
    while idx < cdr.shape[0] : 
        sf.write(join(root_out,train_dev,"cdr","{}_{}_{}_{}_seg_{}.wav".format(dir_dataset,dir_session,id_utt,id_spk,idx)),cdr[idx:idx+len_seg],sr)
        idx += int(len_seg * ratio_shift)


if __name__ == "__main__" : 
    cpu_num = int(cpu_count()/2)

    root_noisy  = "/home/data2/kbh/SPEAR_INPUT"
    root_target = "/home/data2/kbh/SPEAR_TARGET"
    root_cdr = "/home/data2/kbh/SPEAR_CDR"

    root_out = "/home/data2/kbh/SPEAR_cdr_5sec"

    os.makedirs(root_out,exist_ok=True)
    os.makedirs(root_out+"/Train/noisy",exist_ok=True)
    os.makedirs(root_out+"/Train/clean",exist_ok=True)
    os.makedirs(root_out+"/Train/cdr",exist_ok=True)

    os.makedirs(root_out+"/Dev/noisy",exist_ok=True)
    os.makedirs(root_out+"/Dev/clean",exist_ok=True)
    os.makedirs(root_out+"/Dev/cdr",exist_ok=True)

    train_dev = "Train"
    list_doa = []
    list_doa += glob(join(root_noisy,train_dev,"Dataset_1","DOA_sources","*","*.csv"))
    #list_doa += glob(join(root_noisy,train_dev,"Dataset_2","DOA_sources","*","*.csv"))
    #list_doa += glob(join(root_noisy,train_dev,"Dataset_3","DOA_sources","*","*.csv"))
    #list_doa += glob(join(root_noisy,train_dev,"Dataset_4","DOA_sources","*","*.csv"))

    arr = list(range(len(list_doa)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='Train'))


    train_dev = "Dev"
    list_doa = []
    list_doa += glob(join(root_noisy,train_dev,"Dataset_1","DOA_sources","*","*.csv"))
    #list_doa += glob(join(root_noisy,train_dev,"Dataset_2","DOA_sources","*","*.csv"))
    #list_doa += glob(join(root_noisy,train_dev,"Dataset_3","DOA_sources","*","*.csv"))
    #list_doa += glob(join(root_noisy,train_dev,"Dataset_4","DOA_sources","*","*.csv"))

    arr = list(range(len(list_doa)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='Dev'))