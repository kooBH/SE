import os,glob
import librosa as rs
import soundfile as sf
from tqdm.auto import tqdm

input_dir = "/home/data/20250609_eval_simu"
output_root = "/home/data/20250609_eval_simu_1ch"


list_target = glob.glob(os.path.join(input_dir, "**",'*.wav'),recursive=True)

for path in tqdm(list_target):
    prev_dir = os.path.dirname(path).split(input_dir)[-1]
    output_dir = dir_out = output_root + prev_dir
    os.makedirs(dir_out, exist_ok=True)

    filename = os.path.basename(path)
    path_out = output_dir + "/" +filename

    x = rs.load(path, mono=False, sr=16000)[0]
    x = x[0]

    sf.write(path_out, x, 16000)