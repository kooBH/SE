import os, argparse
import glob
from utils.metric import run_metric
import librosa as rs

dir_dns2020      = "/home/data/kbh/DNS2020/test_set/synthetic/no_reverb"

flag_get_score = True

if __name__ == "__main__" :
    list_DNS_noisy = glob.glob(os.path.join(dir_dns2020,"noisy","*.wav"),recursive=True)

    list_DNS=[]
    for path_noisy in list_DNS_noisy :
        token = path_noisy.split("/")[-1]
        token = token.split("_")
        fileid = token[-1].split(".")[0]
        path_clean = os.path.join(dir_dns2020,"clean","clean_fileid_{}.wav".format(fileid))
        list_DNS.append((path_noisy,path_clean))


    PESQ = 0
    # Eval for DNS2020 dev synthetic no reverb
    for pair_data in list_DNS: 
        path_noisy = pair_data[0]
        path_clean = pair_data[1]
        noisy = rs.load(path_noisy,sr=16000)[0]
        clean = rs.load(path_clean,sr=16000)[0]

        val= run_metric(noisy,clean,"PESQ_WB") 
        PESQ += val
        
    PESQ /= len(list_DNS)
    print(PESQ)


