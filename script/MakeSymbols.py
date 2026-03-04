import os,glob,shutil
from tqdm import tqdm

threshold = 3.0
dir_out = "/home/data/DNSDataset_clean_MOS_ge_3"

def makeSymbols(src, dst):
    if not os.path.exists(src):
        print(f"source file {src} does not exist!")
        return
    try :
        os.symlink(src, dst)
    except FileExistsError:
        print(f"symbolic link {dst} already exists.")
    except OSError as e:
        print(f"Failed to create symbolic link from {src} to {dst}: {e}")


list_MOS = glob.glob(os.path.join("DNSMOS_*.txt"))


for path_MOS in list_MOS : 
    filename_MOS = os.path.basename(path_MOS)
    lang = filename_MOS.replace("DNSMOS_","").replace(".txt","")

    dir_lang = os.path.join(dir_out, lang)
    os.makedirs(dir_lang, exist_ok=True)

    with open(path_MOS, 'r') as f:
        lines = f.readlines()

        for line in tqdm(lines) : 
            try : 
                path, mos = line.strip().split()
                if float(mos) < threshold :
                    dst_path = os.path.join(dir_lang, os.path.basename(path))
                    makeSymbols(path, dst_path)
            except :
                print(f"Error line : {line}")
                continue