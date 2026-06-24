import os
import random
from glob import glob
from os.path import join
from typing import Optional

import numpy as np
import librosa as rs
import soundfile as sf
import torch
import warnings
import json
warnings.filterwarnings('ignore')

from scipy.signal import fftconvolve
from Dataset.RIRSimulator import RIRSimulator

def get_list(item, fmt):
    """Glob for files matching `fmt` under one or more root paths."""
    result = []
    if isinstance(item, str):
        result = glob(join(item, '**', fmt), recursive=True)
    elif isinstance(item, list):
        for p in item:
            result += glob(join(p, '**', fmt), recursive=True)
    return result


class DatasetMCSim(torch.utils.data.Dataset):
    """
    Multi-channel simulation dataset.

    Each sample synthesises a noisy multi-channel mixture on-the-fly via
    RIRSimulator, which handles SIR/SNR scaling internally:
      1. Randomly pick 1–max_speakers speech files.
      2. Generate a white-noise diffuse source plus 0–(max_noises-1) directive
         noise files from the noise dataset.
      3. Simulate room acoustics (random shoebox room, random mic/source
         positions) using RIRSimulator.simulate().
      4. Extract the clean reference (anechoic direct path or reverberant).
      5. Level-normalise the mixture to a random target dBFS.

    Returns
    -------
    dict:
        'noisy'   : FloatTensor (n_mics, T)  – full mixture
        'clean'   : FloatTensor (T,)         – clean reference for target speaker
        'mic_pos' : FloatTensor (n_mics, 3)  – mic positions relative to array centre
    """

    def __init__(self, hp, is_train: bool = True):
        self.hp       = hp
        self.is_train = is_train

        if self.is_train : 
            self.sr = hp.data.sr
            self.n_mics    = hp.sim.mic_array.n_mics
            self.ref_mic   = hp.data.ref_mic
            self.len_data  = hp.data.len_data
            self.n_item    = hp.data.n_item
            self.clean_ref = hp.data.clean_ref   # 'anechoic' | 'reverberant'
            self.er_time   = hp.data.early_reflection_time

            self.max_speakers = int(hp.data.max_speakers)
            self.max_noises   = int(getattr(hp.data, 'max_noises', 2))

            # SIR/SNR ranges — support both old (sir_range) and new (SIR_clean) key names
            self.sir_clean_range = (
                list(hp.data.SIR_clean)
                if hasattr(hp.data, 'SIR_clean')
                else list(hp.data.sir_range)
            )
            self.sir_noise_range = (
                list(hp.data.SIR_noise)
                if hasattr(hp.data, 'SIR_noise')
                else list(hp.data.sir_range)
            )
            self.snr_range = list(hp.data.SNR)

            self.target_dB_FS                = hp.data.target_dB_FS
            self.target_dB_FS_floating_value = hp.data.target_dB_FS_floating_value

            # File lists – speech may be any directory/directories with FLAC or WAV
            self.list_clean = (
                get_list(hp.data.speech, '*.flac')
                + get_list(hp.data.speech, '*.wav')
            )
            self.list_noise = (
                get_list(hp.data.noise, '*.wav')
                + get_list(hp.data.noise, '*.flac')
            )

            if not self.list_clean:
                raise FileNotFoundError(
                    f"No speech files found under hp.data.speech = {hp.data.speech!r}"
                )

            self.simulator = RIRSimulator(hp)

            # Optional pre-generated RIR cache
            self.rir_dir = getattr(hp.data, 'rir_dir', None)
            if self.rir_dir:
                self.list_rir_meta = sorted(
                    glob(join(self.rir_dir, 'meta', '*.json'))
                )
                if not self.list_rir_meta:
                    raise FileNotFoundError(
                        f"No RIR meta files found under {self.rir_dir}/meta/"
                    )

            print(
                f"DatasetMCSim[train={is_train}] | n_item={self.n_item} | "
                f"speech={len(self.list_clean)} | noise={len(self.list_noise)} | "
                f"n_mics={self.n_mics} | max_speakers={self.max_speakers} | "
                f"max_noises={self.max_noises} | clean_ref={self.clean_ref}"
                + (f" | rir_dir={self.rir_dir} ({len(self.list_rir_meta)} sets)"
                   if self.rir_dir else " | rir_dir=None (on-the-fly)")
            )
        else : 
            self.sr       = hp.audio.sr
            self.list_noisy = get_list(hp.data.dev.noisy, '*.wav')
            self.list_clean = get_list(hp.data.dev.clean, '*.wav')
            self.meta = get_list(hp.data.dev.meta, '*.json')

            print(
                f"DatasetMCSim[train={is_train}] | n_item={len(self.list_noisy)} | "
            )

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def _load(self, path: str,mono=True) -> np.ndarray:
        """Load audio, resample to self.sr, return mono (T,)."""
        wav,_ = rs.load(path, sr=self.sr, mono=mono)
        return wav.astype(np.float32)
    
    def _load_meta(self,path:str) : 
        with open(path,'r') as f : 
            meta = json.load(f)
        return meta


    def _match_length_1d(self, wav: np.ndarray) -> np.ndarray:
        L = self.len_data
        if len(wav) >= L:
            start = np.random.randint(len(wav) - L + 1)
            return wav[start:start + L]
        return np.pad(wav, (0, L - len(wav)))

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2)))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        if self.is_train:
            while True:
                try:
                    if self.rir_dir:
                        return self._get_sample_from_rir()
                    return self._get_sample()
                except Exception as e:
                    print(f"Warning: exception in DatasetMCSim.__getitem__, retrying. ({e})")
        else:
            return self._get_sample_eval(idx)


    def _load_speech(self) -> np.ndarray:
        cnt = 0
        while True : 
            if cnt > 10:
                raise ValueError("Failed to load non-silent speech after 10 attempts.")
            """Load, trim silence, crop/pad to len_data, and unit-normalise."""
            wav = self._load(random.choice(self.list_clean))
            wav, _ = rs.effects.trim(wav)
            wav = self._match_length_1d(wav)
            if np.max(np.abs(wav)) < 1e-7:
                #raise ValueError("Silent utterance, skipping.")
                cnt += 1
                continue
            return wav / (np.max(np.abs(wav)) + 1e-7)

    def _get_sample(self) -> dict:
        T = self.len_data

        # ---- number of speakers / noise sources for this sample ----
        n_speakers = np.random.randint(1, self.max_speakers + 1)
        n_noises   = np.random.randint(1, self.max_noises + 1)

        # ---- load speech (one per speaker) ----
        cleans = [self._load_speech() for _ in range(n_speakers)]

        # ---- SIR / SNR sampling ----
        # First speaker is the target (reference); additional speakers are interferers.
        clean_SIRs = [0.0] + [
            float(np.random.uniform(self.sir_clean_range[0], self.sir_clean_range[1]))
            for _ in range(n_speakers - 1)
        ]
        SNR = float(np.random.uniform(self.snr_range[0], self.snr_range[1]))

        # ---- build noise list ----
        # Index 0: white-noise diffuse source (always present).
        # Index 1+: directive noise files (0 to max_noises-1 files).
        noise_white = np.random.randn(T).astype(np.float32)
        directive_noises = [
            self._match_length_1d(self._load(random.choice(self.list_noise)))
            for _ in range(n_noises - 1)
        ]
        noises = [noise_white] + directive_noises
        noise_SIRs = [0.0] + [
            float(np.random.uniform(self.sir_noise_range[0], self.sir_noise_range[1]))
            for _ in range(n_noises - 1)
        ]

        # ---- RIR simulation + mixing (SIR/SNR applied internally) ----
        noisy, result = self.simulator.simulate(
            cleans=cleans,
            clean_SIRs=clean_SIRs,
            noise=noises,
            noise_SIRs=noise_SIRs,
            SNR=SNR,
        )
        # noisy : (n_mics, T)

        # ---- clean reference (target speaker only) ----
        if self.clean_ref == 'anechoic':
            # Direct-path only: convolve anechoic speech with RIR up to the peak
            rir_ref = result['rir_target'][self.ref_mic]
            peak    = np.argmax(np.abs(rir_ref))
            clean_ref = fftconvolve(cleans[0], rir_ref[:peak + 1])[:T]
            if len(clean_ref) < T:
                clean_ref = np.pad(clean_ref, (0, T - len(clean_ref)))
            clean_ref = clean_ref.astype(np.float32)
        elif self.clean_ref == "early_reflection":
            rir_ref = result['rir_target'][self.ref_mic]
            clean_ref = self.get_early_reflection(cleans[0], rir_ref, self.sr, er_time=self.er_time)
        else:  # 'reverberant'
            clean_ref = result['reverb_target'][self.ref_mic].copy()

        # Apply the co-normalisation scalar so clean_ref is on the same scale as noisy
        clean_ref = (clean_ref * result['clean_scalar']).astype(np.float32)

        # ---- target dBFS normalisation (noisy and clean jointly scaled) ----
        target_dBFS = float(np.random.uniform(
            self.target_dB_FS - self.target_dB_FS_floating_value,
            self.target_dB_FS + self.target_dB_FS_floating_value,
        ))
        noisy_rms = self._rms(noisy[self.ref_mic]) + 1e-9
        scale = (10 ** (target_dBFS / 20.0)) / noisy_rms
        noisy     = (noisy     * scale).astype(np.float32)
        clean_ref = (clean_ref * scale).astype(np.float32)

        # Clip guard — both arrays share the same denominator
        peak = max(float(np.abs(noisy).max()), float(np.abs(clean_ref).max()))
        if peak > 1.0:
            noisy     = noisy     / peak
            clean_ref = clean_ref / peak

        if np.isnan(noisy).any() or np.isnan(clean_ref).any():
            raise ValueError("NaN in mixed signal, skipping.")

        return {
            'noisy'   : torch.FloatTensor(noisy),              # (n_mics, T)
            'clean'   : torch.FloatTensor(clean_ref),          # (T,)
            'mic_pos' : torch.FloatTensor(result['mic_pos']),  # (n_mics, 3)
            'target_pos' : torch.FloatTensor(result['target_pos']), # (3,)
        }
    
    def _get_sample_from_rir(self) -> dict:
        """Like _get_sample() but loads pre-generated RIRs from self.rir_dir."""
        T = self.len_data

        # Pick a random pre-generated RIR set
        meta_path = random.choice(self.list_rir_meta)
        with open(meta_path) as f:
            meta = json.load(f)

        n_speakers = meta['n_speakers']

        # Load RIRs (paths in meta are relative to rir_dir)
        rirs_target = np.load(os.path.join(self.rir_dir, meta['rir_target']))
        target_azimth = meta['target_azimuth']


        n_interf = np.random.randint(0, self.max_speakers - 1)
        rirs_interf = []
        while len(rirs_interf) < n_interf:
            meta_path = random.choice(self.list_rir_meta)
            with open(meta_path) as f:
                meta_interf = json.load(f)
            interf_azimuth = meta_interf['target_azimuth']

            # Reject Close interferer
            if abs(interf_azimuth - target_azimth) <self.hp.sim.source.min_angle_diff: 
                continue
            else : 
                rirs_interf.append(np.load(os.path.join(self.rir_dir, meta_interf['rir_target'])))
        #rirs_interf = [np.load(os.path.join(self.rir_dir, p)) for p in meta['rir_interf']]
        mic_pos_rel = np.array(meta['mic_pos_rel'], dtype=np.float32)  # (n_mics, 3)

        # Load speech
        cleans = [self._load_speech() for _ in range(n_speakers)]

        # SIR / SNR
        clean_SIRs = [0.0] + [
            float(np.random.uniform(self.sir_clean_range[0], self.sir_clean_range[1]))
            for _ in range(n_speakers - 1)
        ]
        SNR = float(np.random.uniform(self.snr_range[0], self.snr_range[1]))

        # Diffuse white noise only (no gpuRIR needed)
        noise_white = np.random.randn(T).astype(np.float32)

        noisy, result = self.simulator.simulate_from_rir(
            cleans=cleans,
            clean_SIRs=clean_SIRs,
            rirs_target=rirs_target,
            rirs_interf=rirs_interf,
            mic_pos_rel=mic_pos_rel,
            noise=[noise_white],
            noise_SIRs=[0.0],
            SNR=SNR,
        )

        # Clean reference extraction (same logic as _get_sample)
        if self.clean_ref == 'anechoic':
            rir_ref = rirs_target[self.ref_mic]
            peak    = np.argmax(np.abs(rir_ref))
            clean_ref = fftconvolve(cleans[0], rir_ref[:peak + 1])[:T]
            if len(clean_ref) < T:
                clean_ref = np.pad(clean_ref, (0, T - len(clean_ref)))
            clean_ref = clean_ref.astype(np.float32)
        elif self.clean_ref == 'early_reflection':
            rir_ref   = rirs_target[self.ref_mic]
            clean_ref = self.get_early_reflection(cleans[0], rir_ref, self.sr, er_time=self.er_time)
        else:  # 'reverberant'
            clean_ref = result['reverb_target'][self.ref_mic].copy()

        clean_ref = (clean_ref * result['clean_scalar']).astype(np.float32)

        # Target dBFS normalisation
        target_dBFS = float(np.random.uniform(
            self.target_dB_FS - self.target_dB_FS_floating_value,
            self.target_dB_FS + self.target_dB_FS_floating_value,
        ))
        noisy_rms = self._rms(noisy[self.ref_mic]) + 1e-9
        scale = (10 ** (target_dBFS / 20.0)) / noisy_rms
        noisy     = (noisy     * scale).astype(np.float32)
        clean_ref = (clean_ref * scale).astype(np.float32)

        peak = max(float(np.abs(noisy).max()), float(np.abs(clean_ref).max()))
        if peak > 1.0:
            noisy     = noisy     / peak
            clean_ref = clean_ref / peak

        if np.isnan(noisy).any() or np.isnan(clean_ref).any():
            raise ValueError("NaN in mixed signal, skipping.")

        target_pos = np.array(meta['target_pos'], dtype=np.float32).reshape(1, 3)  # (1, 3) for broadcasting

        return {
            'noisy'      : torch.FloatTensor(noisy),
            'clean'      : torch.FloatTensor(clean_ref),
            'mic_pos'    : torch.FloatTensor(mic_pos_rel),
            'target_pos' : torch.FloatTensor(target_pos),
        }

    def _get_sample_eval(self, idx: int) -> dict:
        noisy_path = self.list_noisy[idx]
        clean_path = self.list_clean[idx]
        meta_path  = self.meta[idx]

        noisy = self._load(noisy_path,mono=False)
        clean = self._load(clean_path)
        meta = self._load_meta(meta_path)

        target_pos = np.array(meta['target_pos'], dtype=np.float32).reshape(1, 3)  # (1, 3) for broadcasting

        return {
            'noisy'   : torch.FloatTensor(noisy),              # (T,)
            'clean'   : torch.FloatTensor(clean),              # (T,)
            'mic_pos' : torch.FloatTensor(meta["mic_pos"]),                                  # No mic positions for eval/dev
            'target_pos' :  torch.FloatTensor(target_pos),                            # No target position for eval/dev
        }

    def __len__(self) -> int:
        if self.is_train :
            return self.n_item
        else :
            return len(self.list_noisy)
    
    def get_early_reflection(self,clean: np.ndarray, rir: np.ndarray,
                         sr: int, er_time: float) -> np.ndarray:
        """
        Convolve `clean` with the first `er_time` seconds of `rir`.

        Parameters
        ----------
        clean   : (T,)        anechoic speech
        rir     : (rir_len,)  full RIR for one microphone
        sr      : sample rate
        er_time : early-reflection window in seconds

        Returns
        -------
        (T,) early-reverberant signal
        """
        n_er = max(1, int(er_time * sr))
        T = len(clean)
        out = fftconvolve(clean, rir[:n_er])[:T]
        if len(out) < T:
            out = np.pad(out, (0, T - len(out)))
        return out.astype(np.float32)
    



# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from utils.hparams import HParam

    hp = HParam(
        './config/data/train_1_linear_6.yaml'
    )
    ds = DatasetMCSim(hp, is_train=True)

    for i in range(5):
        sample = ds[i]
        print(
            f"[{i}] noisy={tuple(sample['noisy'].shape)}  "
            f"clean={tuple(sample['clean'].shape)}  "
            f"noisy_max={sample['noisy'].abs().max():.4f}"
        )
