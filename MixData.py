import os,glob,sys
import librosa as rs
import soundfile as sf
import numpy as np
import random
from tqdm.auto import tqdm

import data.Mixer as m
from utils.hparams import HParam

def match_length(x,len_data,idx_start=None) : 
    if len(x) > len_data : 
        left = len(x) - len_data
        if idx_start is None :
            idx_start = np.random.randint(left)
        x = x[idx_start:idx_start+len_data]
    elif len(x) < len_data : 
        shortage = len_data - len(x) 
        x = np.pad(x,(0,shortage))
    return x, idx_start

def search_file(root) :
    formats = ["*.wav", "*.flac", "*.mp3"]
    list_audio = []

    for fmt in formats:
        list_audio.extend(glob.glob(os.path.join(root, "**", fmt), recursive=True))
    return list_audio

if __name__ == "__main__":
    #hp = HParam("config/data/VD.yaml")
    #hp = HParam("config/data/DNS.yaml")
    #hp = HParam("config/data/LibriSpeechTest.yaml")
    hp = HParam("config/data/LibriSpeechDev.yaml")

    list_clean = search_file(hp.root_clean)
    list_noise = search_file(hp.root_noise)
    list_RIR = search_file(hp.RIR)

    save_dir = os.path.join(hp.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"noisy"), exist_ok=True)
    os.makedirs(os.path.join(save_dir,"noise"), exist_ok=True)
    os.makedirs(os.path.join(save_dir,"clean"), exist_ok=True)

    print(f"Clean : {len(list_clean)} | Noise : {len(list_noise)}")

    print(f"SNR : {hp.SNR}")
    print(f"Reverb : {hp.Reverb}")
    print(f"Scale : {hp.Scale}")

    random.shuffle(list_clean)

    n_item = min(hp.n_item, len(list_clean))

    for i in tqdm(range(n_item)) :
        if hp.max_len_data is not None :
            while True :
                clean_path = random.choice(list_clean)
                clean, sr = rs.load(clean_path, sr=hp.sr)
                if len(clean) >= hp.max_len_data:
                    break
        noise_path = random.choice(list_noise)
        noise, sr = rs.load(noise_path, sr=hp.sr)

        clean , _ = match_length(clean, hp.len_data)
        noise , _ = match_length(noise, hp.len_data)

        rir = None
        if hp.Reverb.use : 
            if random.random() < hp.Reverb.prob:
                rir_path = random.choice(list_RIR)
                rir, sr = rs.load(rir_path, sr=hp.sr)

        noisy, clean, noise = m.mix(
            clean,noise,
            rir = rir,
            target_dB_FS = hp.Scale.target_dB_FS,
            target_dB_FS_floating_value = hp.Scale.target_dB_FS_floating_value,
            scale_method = hp.Scale.method,
            range_SNR=hp.SNR,
            deverb_clean=hp.Reverb.deverb_clean,
            clean_rir_len=hp.Reverb.clean_rir_len,
            use_spec_augmentation = False,
        )

        base_name = os.path.basename(clean_path)
        base_name = base_name.split(".")[0]

        sf.write(os.path.join(save_dir,"noisy",f"noisy_{base_name}.wav"), noisy, sr)
        sf.write(os.path.join(save_dir,"clean",f"clean_{base_name}.wav"), clean, sr)
        sf.write(os.path.join(save_dir,"noise",f"noise_{base_name}.wav"), noise, sr)




