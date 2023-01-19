import os,glob
import librosa as rs
import soundfile as sf

# utils
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

dir_input = "/home/data/kbh/DNS-Challenge/datasets_fullband/"
dir_output = "/home/data/kbh/DNS-Challenge-16kHz/"
sr = 16000

list_target = glob.glob(os.path.join(dir_input,"**","*.wav"),recursive=True)

print("data : {}".format(len(list_target)))

def convert(idx) : 
    # Path Management
    path = list_target[idx]

    name_item = path.split('/')[-1]
    name_target = path.split('/')[-1]
    id_target = name_target.split('.')[0]

    path_after_root = path.split(dir_input)[1]
    path_before_name = path_after_root.split(name_item)[0]

    ## see : https://docs.python.org/3/library/os.path.html#os.path.join
    if path_before_name == "/" : 
        path_before_name = ""

    x = rs.load(path,sr=sr,res_type="fft")[0]

    os.makedirs(os.path.join(dir_output,path_before_name),exist_ok=True)  

    sf.write(os.path.join(dir_output,path_before_name,"{}".format(name_item)),x,sr)
    

if __name__=='__main__': 
    cpu_num = cpu_count()

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(convert, arr), total=len(arr),ascii=True,desc='Covnerting to 16kHz'))