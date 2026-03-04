import torch
import numpy as np
import librosa as rs
import os,glob
from tqdm import tqdm
import signal

class DNSMOS_singleton(object):
    def __new__(cls, primary_model_path,sr=16000):
        if not hasattr(cls,'instance'):
            import onnxruntime as ort
            cls.onnx_sess = ort.InferenceSession(primary_model_path,providers=["CPUExecutionProvider"])
            cls.sr = sr
            cls.instance = super(DNSMOS_singleton, cls).__new__(cls)
            cls.INPUT_LENGTH = 9.01
            print("metric.py::DNSMOS initialized")
        # recycle
        else :
            pass
        return cls.instance
    
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = rs.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (rs.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly
    
    def __call__(self, aud, input_fs):
        fs = self.sr
        if input_fs != fs:
            audio = rs.resample(aud, orig_sr=input_fs, target_sr=self.sr)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.INPUT_LENGTH*fs)

        if len(audio) == 0:
            return 0

        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - self.INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            oi = {'input_1': input_features}
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,False)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        return clip_dict
    
def DNSMOS(estim,target,fs=16000, ret_all = False):
    model = DNSMOS_singleton("sig_bak_ovr.onnx",fs)
    clip_dict = model(estim, input_fs=fs)
    if type(clip_dict) is int :
        return -1
    if ret_all : 
        return [clip_dict['OVRL'],clip_dict['SIG'],clip_dict['BAK']]
    else :
        return clip_dict["OVRL"]
    
if __name__ == "__main__" :

    dir_in = "/home/data/DNS-Challenge-16kHz/datasets_fullband/clean_fullband"
    list_dir = glob.glob(os.path.join(dir_in, "*"))

    for item_dir in list_dir : 
        name_dir = os.path.basename(item_dir)

        # skip italian
        if name_dir.startswith("italian") :
            pass
        else: 
            continue

        list_target  = glob.glob( os.path.join(item_dir,"**","*.wav"), recursive=True)
        sum_MOS = 0.0
        cnt = 1
        cnt_MOS_lower_3 = 0
        list_MOS = []
        pbar = tqdm(list_target)
        pbar.set_description(f"{name_dir}")

        with open(f"DNSMOS_{name_dir}.txt","w") as f :
            for path in pbar :
                mos = 0.0
                name_file = os.path.basename(path)
                #pbar.set_postfix(_file=name_file)
                x = rs.load(path, sr=16000)[0]
                mos = DNSMOS( x , None, fs=16000)
                if mos < 0 :
                    print("Error for file : ", path)
                    continue
                sum_MOS += mos
                if mos < 3.0 :
                    cnt_MOS_lower_3 += 1
                    list_MOS.append((path,mos))
                cnt+=1
                f.write(f"{path}\t{mos:.2f}\n")
                pbar.set_postfix(_file=name_file, avg=f"{sum_MOS/cnt:.2f}", ratio = f"{cnt_MOS_lower_3/cnt:.2f}")
            avg_MOS = sum_MOS / len(list_target)

            print(f"Directory : {name_dir}")
            print("Total files : ", len(list_target))
            print("Average DNSMOS : ", avg_MOS)
            print("Number of files with DNSMOS lower than 3.0 : ", cnt_MOS_lower_3)