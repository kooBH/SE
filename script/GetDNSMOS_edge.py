import numpy as np
import librosa as rs
import os,glob
import shutil
from tqdm import tqdm

dir_samples = "../data/MOS"
os.makedirs(dir_samples,exist_ok=True)

list_stats = glob.glob(os.path.join("DNSMOS_*.txt"))

for item_dict in list_stats :
    print(item_dict)
    lang = item_dict.replace("DNSMOS_","").replace(".txt","")

    max_mos = -1.0
    min_mos = 6.0
    max_path = ""
    min_path = ""
    with open(item_dict,"r") as f :
        while True :
            line = f.readline()
            if not line :
                break
            name_file, mos = line.strip().split()
            mos = float(mos)

            if mos > max_mos :
                max_mos = mos
                max_path = name_file
            if mos < min_mos :
                min_mos = mos
                min_path = name_file

    shutil.copy(max_path, os.path.join(dir_samples,f"DNSMOS_best_{lang}_MOS{max_mos:.2f}.wav"))
    shutil.copy(min_path, os.path.join(dir_samples,f"DNSMOS_worst_{lang}_MOS{min_mos:.2f}.wav"))

    break

