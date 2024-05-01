import os, argparse
import glob
import torch
import librosa as rs
import soundfile as sf
from utils.hparams import HParam
from common import get_model
from tqdm.auto import tqdm


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                    help="default configuration")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--dir_in',"-i", type=str, required=True)
    parser.add_argument('--dir_out',"-o", type=str, required=True)
    parser.add_argument('--device',"-d", type=str, required=True)
    args = parser.parse_args()

    hp = HParam(args.config,args.default,merge_except=["architecture"])
    print("NOTE::Loading configuration : "+args.config)
    device = args.device

    os.makedirs(args.dir_out,exist_ok=True)

    list_data = glob.glob(os.path.join(args.dir_in,"**","*.wav"),recursive=True)

    model = get_model(hp,device)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))
    model.eval()

    for path in tqdm(list_data) : 
        base_dir = os.path.dirname(path)
        base_dir = base_dir.replace(args.dir_in,args.dir_out)

        data = rs.load(path,sr=hp.data.sr)[0]
        data = torch.unsqueeze(torch.from_numpy(data),0).to(device)
        estim = model(data)

        estim = estim.cpu().detach().numpy()
        estim = estim[0]

        sf.write(os.path.join(base_dir,os.path.basename(path)),estim,hp.data.sr)

