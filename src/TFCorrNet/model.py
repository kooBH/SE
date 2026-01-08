import sys
sys.path.append('../')

import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

# import sys
# sys.path.append('/home/Uihyeop/nas_Uihyeop/NN_SE/TPMCN_CSS')
# from models.TPMCN_tiny_RM_v1.modules.module import *

from utils.decorators import *
from .modules.module import *
import numpy as np
from utils import util_stft
from TF_CorrNet_SE.TF_CorrNet_2_stage_v1 import loss

# @logger_wraps()
class TF_CorrNet(torch.nn.Module):
    def __init__(self, 
                 ISCM_embedding: dict,
                 multi_path_block: dict,
                 relative_positional_encoding: dict,
                 N_repeat: int, 
                 seq_down_factor: int,
                 mask_estimator: dict):
        super().__init__()

        self.iscm_embed = ISCMembedding(**ISCM_embedding)
        
        self.pos_emb_f = RelativePositionalEncoding(**relative_positional_encoding["spectral_module"])
        self.pos_emb_t = RelativePositionalEncoding(**relative_positional_encoding["channel_module"])
        self.pos_emb_c = RelativePositionalEncoding(**relative_positional_encoding["low_rank_module"])

        self.TP_blocks = torch.nn.Sequential(*[Multi_Path_Block(**multi_path_block) for _ in range(N_repeat)])

        self.mask_estim = MaskEstimator(**mask_estimator)

        self.seq_down_factor = seq_down_factor

        
    def forward(self, x):
        # x : (B), T, F, 2M 
        if len(x.shape) == 3: # When No Batch Dimension
            x = x.unsqueeze(0)

        #x = self.iscm_embed(x)
        pos_k = []
        B, T, C, F = x.shape
        pos_seq = torch.arange(0, T//self.seq_down_factor).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k.append(self.pos_emb_c(pos_seq)[0])
        pos_seq = torch.arange(0, F//self.seq_down_factor).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k.append(self.pos_emb_f(pos_seq)[0])
        pos_seq = torch.arange(0, T//self.seq_down_factor).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k.append(self.pos_emb_t(pos_seq)[0])
        
        for block in self.TP_blocks:
            x = block(x, pos_k)

        m = self.mask_estim(x) # B, M, W, F, T, 2
        return m
    
class TF_CorrNet_helper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TF_CorrNet(**config["TF_CorrNet"])

        self.stft = util_stft.STFT(**config['stft'])
        self.istft = util_stft.iSTFT(**config['stft'])
        self.filtering_1st = loss.filtering

    def forward(self, x):
        mixture_stft = self.stft(x, cplx=True).unsqueeze(1)
        mixture_stft = mixture_stft.permute(0, 3, 2, 1).contiguous()
        x_input = torch.cat([torch.real(mixture_stft), torch.imag(mixture_stft)],dim=-1)
        mixture_stft = mixture_stft.permute(0, 3, 2, 1).contiguous()

        y = self.model(x_input)
        mask = torch.complex(y[...,0], y[...,1]) # masks = [torch.complex(mask[...,0], mask[...,1]) for mask in masks]
        src_estim = self.filtering_1st(mask, mixture_stft) # src_estim = [self.filtering_1st(mask, mixture_stft) for mask in masks]
        # spk_mvdr = [self.mvdr(x=mixture_stft, src=src, transpose=False) for src in src_estim]

        y = self.istft(src_estim, cplx=True)
        return y      

if __name__ == "__main__":

    import librosa as rs
    import soundfile as sf
    import argparse
    from utils.hparams import HParam
    from glob import glob
    import os
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    args = parser.parse_args()

    hp = HParam(args.config,args.config,merge_except=["architecture"])

    model = TF_CorrNet_helper(hp).to("cuda:0")
    model.eval()
    
    model.load_state_dict(torch.load('mpANC/model_largeV2.pt', weights_only=True))

    # input_dir = "/mnt/hdisk1/LibriTest"
    # output_dir = "/mnt/hdisk1/LibriTest_small"

    # wavList = glob(os.path.join(input_dir, "**", "*.wav"),recursive=True)

    # for wav in wavList:
    #     x = rs.load(wav,sr=16000)[0]
    #     x = torch.tensor(x).unsqueeze(0)

    #     y = model(x)
    #     y = y.squeeze(0).detach().numpy()

    #     rel_path = os.path.relpath(wav, input_dir)
    #     rel_dir = os.path.dirname(rel_path)
    #     base_name = os.path.splitext(os.path.basename(rel_path))[0]
    #     new_filename = f"{base_name}.wav"

    #     out_path = os.path.join(output_dir, rel_dir, new_filename)
    #     os.makedirs(os.path.dirname(out_path), exist_ok=True)

    #     sf.write(out_path, y, 16000)


    x, sr = rs.load("mpANC/TV_16kHz_mono.wav",sr=16000, duration=1.0)
    x = torch.tensor(x).unsqueeze(0).to('cuda:0')
    #x = x.permute(0, 2, 1)

    # 워밍업
    for _ in range(5):  
        _ = model(x)

    start = time.time()
    y = model(x)
    end = time.time()
    y = y.squeeze(0).detach().cpu().numpy()

    print(f"Inference time: {(end - start) * 1000:.3f} ms")
    print(f"Audio length: {len(x[0]) / sr:.2f} seconds")

    sf.write("mpANC/output.wav",y,16000)
