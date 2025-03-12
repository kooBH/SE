import os
from os.path import join
from glob import glob
import torch
import librosa as rs
import soundfile as sf
import numpy as np
import random
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
from scipy import signal
warnings.filterwarnings('ignore')

def get_list(item,format) : 
    list_item = []
    if type(item) is str :
        list_item = glob(join(item,"**",format),recursive=True)
    elif type(item) is list :
        for i in item : 
            list_item += glob(join(i,"**",format),recursive=True)
    return list_item

class DatasetDNS(torch.utils.data.Dataset):
    def __init__(self,hp,is_train=True):
        self.hp = hp
        self.is_train = is_train

        if is_train : 
            self.list_clean = get_list(hp.data.clean,"*.wav")
            self.list_noise = get_list(hp.data.noise,"*.wav")
            self.list_RIR = get_list(hp.data.RIR,"*.wav")
        else :
            self.list_noisy = glob(join(hp.data.dev.root,"*","noisy","*.wav"),recursive=True)
            self.list_RIR   = None
            self.list_clean = None

            self.eval={}
            self.eval["with_reverb"]={}
            self.eval["no_reverb"]={}

            self.eval["with_reverb"] = glob(join(hp.data.dev.root,"with_reverb","noisy","*.wav"),recursive=True)

            self.eval["no_reverb"] = glob(join(hp.data.dev.root,"no_reverb","noisy","*.wav"),recursive=True)

        self.range_SNR = hp.data.SNR
        self.target_dB_FS = hp.data.target_dB_FS
        self.target_dB_FS_floating_value = hp.data.target_dB_FS_floating_value

        self.len_data = hp.data.len_data
        self.n_item = hp.data.n_item

        self.sr = hp.data.sr

        if is_train : 
            print("DatasetDNS[train:{}] | len : {} | clean {} | noise : {} | RIR : {}".format(is_train,self.n_item,len(self.list_clean),len(self.list_noise),len(self.list_RIR)))
        else :
            print("DatasetDNS[train:{}] | len : {} | noisy : {}".format(is_train,self.n_item,len(self.list_noisy)))

    def match_length(self,wav,idx_start=None) : 
        if len(wav) > self.len_data : 
            left = len(wav) - self.len_data
            if idx_start is None :
                idx_start = np.random.randint(left)
            wav = wav[idx_start:idx_start+self.len_data]
        elif len(wav) < self.len_data : 
            shortage = self.len_data - len(wav) 
            wav = np.pad(wav,(0,shortage))
        return wav, idx_start

    @staticmethod
    def norm_amplitude(y, scalar=None, eps=1e-6):
        if not scalar:
            scalar = np.max(np.abs(y)) + eps

        return y / scalar, scalar

    @staticmethod
    def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 10) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    @staticmethod
    def is_clipped(y, clipping_threshold=0.999):
        return any(np.abs(y) > clipping_threshold)

    def mix(self,clean,noise,rir,eps=1e-7):
        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            self.target_dB_FS - self.target_dB_FS_floating_value,
            self.target_dB_FS + self.target_dB_FS_floating_value
        )
        
        # skip mixing if clean is silent
        if np.sum(np.abs(clean)) < 1e-13:
            noise, _, noise_scalar = self.tailor_dB_FS(noise, noisy_target_dB_FS)

            return noise, clean, noise
        
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            if self.hp.data.deverb_clean :
                peak = np.argmax(rir)
                clean_peak  = signal.fftconvolve(clean,rir[:peak])[:len(clean)]
            clean = signal.fftconvolve(clean, rir)[:len(clean)]


        clean, _ = self.norm_amplitude(clean)
        clean, _, _ = self.tailor_dB_FS(clean, self.target_dB_FS)
        clean_rms = (clean ** 2).mean() ** 0.5

        noise, _ = self.norm_amplitude(noise)
        noise, _, _ = self.tailor_dB_FS(noise, self.target_dB_FS)
        noise_rms = (noise ** 2).mean() ** 0.5

        SNR = np.random.randint(
            self.range_SNR[0],self.range_SNR[1]
        )

        snr_scalar = clean_rms / (10 ** (SNR / 10)) / (noise_rms + eps)
        noise *= snr_scalar
        noisy = clean + noise

        if rir is not None:
            if self.hp.data.deverb_clean : 
                clean = clean_peak

        # rescale noisy RMS
        noisy, _, noisy_scalar = self.tailor_dB_FS(noisy, noisy_target_dB_FS)
        clean *= noisy_scalar

        if self.is_clipped(noisy) :
            noisy_scalar = np.max(np.abs(noisy)) / (0.99 - eps)  # same as divide by 1
            noisy = noisy / noisy_scalar
            clean = clean / noisy_scalar

        if self.is_clipped(clean) : 
            clean_scalar = np.max(np.abs(clean)) / (0.99 - eps)  # same as divide by 1
            noisy = noisy / clean_scalar
            clean = clean / clean_scalar

        return noisy, clean, noise
    
    def get_clean_dev(self,path_noisy):
        path_after_root = path_noisy.split(self.hp.data.dev.root)[-1]
        dev_type = path_after_root.split("/")[0]

        fid = path_noisy.split("_")[-1]
        fid = fid.split(".")[0]

        path_clean = os.path.join(self.hp.data.dev.root,dev_type,"clean","clean_fileid_{}.wav".format(fid))

        return path_clean
    
    def mix_clean(self,n_spk):
        cur_spk = 0
        clean_pool = None
        path = []
        while cur_spk < n_spk : 
            path_clean = random.choice(self.list_clean)
            clean = rs.load(path_clean,sr=self.sr)[0]
            clean, trim_idx = rs.effects.trim(clean)

            # norm clean
            #clean = clean / np.max(np.abs(clean))


            # main speaker 
            if cur_spk == 0 :
                clean,_ = self.match_length(clean)

                clean_pool = clean
            # overlaped speaker
            else : 
                # Start point addjust
                space = self.len_data * np.random.uniform(self.hp.data.range_multitalk[0],self.hp.data.range_multitalk[1])

                clean = np.pad(clean,(int(space),0))
                clean,_ = self.match_length(clean,idx_start = 0)
                if np.sum(np.abs(clean)) < 1e-7 : 
                    continue
                clean_pool += clean
            path.append(path_clean)

            cur_spk += 1
        # norm clean_pool
        clean_pool_t = clean_pool / np.max(np.abs(clean_pool))

        if np.isnan(clean_pool_t).any() :
            #print("Dataset::mix_clean:: There is nan is clean_pool : {} {}".format(path,trim_idx))
            clean_pool_t = np.nan_to_num(clean_pool_t)

        return clean_pool_t,path

    def residual_clean(self, clean, noise):
        SNR = np.random.uniform(self.hp.data.residual_clean.SNR[0],self.hp.data.residual_clean.SNR[1])

        clean_rms = (clean ** 2).mean() ** 0.5
        noise_rms = (noise ** 2).mean() ** 0.5
        snr_scalar = clean_rms / (10 ** (SNR / 10)) / (noise_rms + 1e-7)
        noise *= snr_scalar

        return clean

    def __getitem__(self, idx):

        if self.is_train : 
            n_spk = np.random.choice(len(self.hp.data.prob_spk),1,p=self.hp.data.prob_spk)[0]

            # sample clean
            if n_spk == 0 :
                flag_clean = True
                clean = np.zeros(self.len_data)
            else :
                flag_clean = False

            while not flag_clean :

                try : 
                    clean,path_clean = self.mix_clean(n_spk)
                except Exception  as e :
                    print("Warning::Exception occured resample data", e)
                    continue

                if np.isnan(clean).any() :
                    print("Dataset::getitem:: There is nan is clean : {}".format(path_clean))
                    continue

                flag_clean = True


            flag_noise = False

            # sample noise
            while not flag_noise :
                path_noise = random.choice(self.list_noise)
                noise = rs.load(path_noise,sr=self.sr)[0]

                flag_noise = True

            if self.hp.data.use_RIR : 
                if self.hp.data.RIR_prob > random.random() :
                    # sample RIR
                    path_RIR = random.choice(self.list_RIR)
                    RIR = rs.load(path_RIR,sr=self.sr)[0]
                else :
                    RIR = None
            else :
                RIR = None

            ## Length Match
            clean,_ = self.match_length(clean)
            noise,_ = self.match_length(noise)

            # mix 
            noisy,clean,noise = self.mix(clean,noise,RIR)

            # clean with residual noise
            if self.hp.data.residual_clean.use : 
                clean = self.residual_clean(clean,noise)

            #print("{} | {:.4f} {:.4f} | {:.4f} {:.4f} | {:.4f} {:.4f}".format(idx,np.sum(clean),np.max(np.abs(clean)),np.sum(noise),np.max(np.abs(noise)),np.sum(noisy),np.max(np.abs(noisy))))

        else :
            path_noisy = self.list_noisy[idx]
            path_clean = self.get_clean_dev(path_noisy)
            
            clean = rs.load(path_clean,sr=self.sr)[0]
            noisy = rs.load(path_noisy,sr=self.sr)[0]

            clean,idx_start = self.match_length(clean)
            noisy,_ = self.match_length(noisy,idx_start)

        data = {"noisy":noisy,"clean":clean}
        return  data

    def __len__(self):
        if self.is_train : 
            return self.n_item
        else :
            return len(self.list_noisy)
        
    def get_eval(self,idx) : 

        path_reverb = self.eval["with_reverb"][idx]
        path_no_reverb = self.eval["no_reverb"][idx]

        path_clean_reverb = self.get_clean_dev(path_reverb)
        path_clean_no_reverb = self.get_clean_dev(path_no_reverb)

        return [path_reverb,path_clean_reverb],[path_no_reverb,path_clean_no_reverb]

## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("../")
    from utils.hparams import HParam
    hp = HParam("../../config/mpSEv2/v138.yaml","../../config/mpSEv2/default.yaml")

    db = DatasetDNS(hp,is_train=True)

    def check(db) : 
        for i in range(10000): 
            sample_clean = db[i]["clean"]
            sample_noisy = db[i]["noisy"]

            abs_clean = np.abs(sample_clean)
            abs_noisy = np.abs(sample_noisy)

            if np.isnan(sample_clean).any() : 
                import pdb; pdb.set_trace()

            #print("shape {:.3f} {:.3f} | avg {:.3f} {:.3f} | sum {:.3f} {:.3f} | max {:.3f} {:.3f}".format(sample_clean.shape[0], sample_noisy.shape[0], np.mean(abs_clean), np.mean(abs_noisy), np.sum(abs_clean), np.sum(abs_noisy), np.max(abs_clean), np.max(abs_noisy)))

    check(db)
    import pdb; pdb.set_trace()
  




