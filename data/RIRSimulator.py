import numpy as np
from scipy.signal import fftconvolve
from typing import List, Optional
import librosa as rs
import soundfile as sf

import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)

def pre_emphasis(signal, coeff=0.99):
    # Standard pre-emphasis filter to boost high frequencies
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def _rms(signal, vad=False):
    if vad:
        intervals = rs.effects.split(signal, top_db=30) 
        signal = np.concatenate([signal[start:end] for start, end in intervals])
    signal = pre_emphasis(signal)

    signal_rms = (signal** 2).mean() ** 0.5
    return signal_rms

class RIRSimulator:

    """
    Simulates multi-channel Room Impulse Responses on-the-fly using
    gpuRIR (image source method, shoebox room).

    Supports linear and circular microphone array geometries.
    Room dimensions and RT60 are sampled uniformly from configured ranges
    each call to simulate().
    """

    def __init__(self, hp):

        self.sr  = hp.data.sr
        self.cfg = hp.sim

    # ------------------------------------------------------------------
    # Room / geometry helpers
    # ------------------------------------------------------------------

    def _random_room_dim(self) -> np.ndarray:
        lo = self.cfg.room.min_dim
        hi = self.cfg.room.max_dim
        return np.array([np.random.uniform(lo[i], hi[i]) for i in range(3)])

    def _mic_positions(self, center: np.ndarray) -> np.ndarray:
        """
        Build mic positions around `center`.

        Returns
        -------
        positions : (3, n_mics)
        """
        mc = self.cfg.mic_array
        n  = mc.n_mics

        if mc.type == 'linear':
            half    = (n - 1) * mc.spacing / 2
            offsets = np.linspace(-half, half, n)
            pos     = np.tile(center[:, None], (1, n)).astype(float)
            pos[0] += offsets

        elif mc.type == 'circular':
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos    = np.tile(center[:, None], (1, n)).astype(float)
            pos[0] += mc.radius * np.cos(angles)
            pos[1] += mc.radius * np.sin(angles)

        elif mc.type == 'dynamic':
            pos = self._dynamic_mic_positions(center, n)

        else:
            raise ValueError(f"Unknown mic_array type: '{mc.type}'. "
                             "Choose 'linear', 'circular', or 'dynamic'.")
        return pos

    def _dynamic_mic_positions(
        self, center: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Sample `n` mic positions randomly around `center` in the XY plane,
        subject to pairwise distance constraints.

        Constraints (from cfg.mic_array):
            min_pair_dist : every pair of mics must be at least this far apart
            max_pair_dist : every pair of mics must be no farther than this

        Falls back to a uniform circular layout at the mid-radius if no valid
        configuration is found within 1000 trials.

        Returns
        -------
        positions : (3, n_mics)
        """
        mc      = self.cfg.mic_array
        min_d   = float(mc.min_pair_dist)
        max_d   = float(mc.max_pair_dist)
        radius  = max_d / 2.0   # bounding disk radius for candidate draws

        for _ in range(1000):
            xy = np.random.uniform(-radius, radius, (n, 2))
            # compute all pairwise distances
            diffs = xy[:, None, :] - xy[None, :, :]          # (n, n, 2)
            dists = np.sqrt((diffs ** 2).sum(-1))             # (n, n)
            # mask out self-distances
            mask = ~np.eye(n, dtype=bool)
            pair_dists = dists[mask]
            if pair_dists.min() >= min_d and pair_dists.max() <= max_d:
                pos     = np.tile(center[:, None], (1, n)).astype(float)
                pos[0] += xy[:, 0]
                pos[1] += xy[:, 1]
                return pos

        # Fallback: evenly spaced circle at mid-radius
        r      = (min_d + max_d) / 4.0
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos    = np.tile(center[:, None], (1, n)).astype(float)
        pos[0] += r * np.cos(angles)
        pos[1] += r * np.sin(angles)
        return pos

    def _random_source_pos(
        self,
        room_dim: np.ndarray,
        array_center: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a source position inside the room that satisfies the
        min/max distance constraints from the array center.
        Falls back to a clipped position if no valid sample is found
        within 200 trials.
        """
        src = self.cfg.source
        margin = 0.3

        for _ in range(200):
            x = np.random.uniform(margin, room_dim[0] - margin)
            y = np.random.uniform(margin, room_dim[1] - margin)
            z = np.random.uniform(src.height_range[0], src.height_range[1])
            z = min(z, room_dim[2] - margin)
            pos  = np.array([x, y, z])
            dist = np.linalg.norm(pos[:2] - array_center[:2])
            if src.min_distance <= dist <= src.max_distance:
                return pos

        # Fallback: project onto a valid distance
        angle = np.random.uniform(0, 2 * np.pi)
        d     = np.random.uniform(src.min_distance, src.max_distance)
        pos   = array_center.copy()
        pos[0] = np.clip(array_center[0] + d * np.cos(angle), margin, room_dim[0] - margin)
        pos[1] = np.clip(array_center[1] + d * np.sin(angle), margin, room_dim[1] - margin)
        pos[2] = np.clip(
            np.random.uniform(src.height_range[0], src.height_range[1]),
            margin, room_dim[2] - margin,
        )
        return pos

    # ------------------------------------------------------------------
    # Multi-speaker helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _azimuth(pos: np.ndarray, center: np.ndarray) -> float:
        """Azimuth angle (degrees) of pos relative to center in the XY plane."""
        return float(np.degrees(np.arctan2(pos[1] - center[1], pos[0] - center[0])))

    @staticmethod
    def _angle_diff(az1: float, az2: float) -> float:
        """Shortest angular separation (degrees) between two azimuths."""
        diff = abs(az1 - az2) % 360
        return min(diff, 360 - diff)

    def _sample_speaker_positions(
        self,
        n_speakers: int,
        room_dim: np.ndarray,
        array_center: np.ndarray,
        min_angle_diff: float,
    ) -> list:
        """
        Sample `n_speakers` source positions subject to a minimum azimuth
        angle constraint between every pair of speakers.

        Each position still satisfies the usual distance constraints from
        `_random_source_pos`.  If no valid placement is found within 300
        trials, the last sampled position is accepted unconditionally.
        """
        positions: list = []
        azimuths:  list = []

        for _ in range(n_speakers):
            pos = None
            for _ in range(300):
                candidate = self._random_source_pos(room_dim, array_center)
                az = self._azimuth(candidate, array_center)
                if all(self._angle_diff(az, a) >= min_angle_diff for a in azimuths):
                    pos = candidate
                    az_accepted = az
                    break
            if pos is None:   # fallback – accept last candidate without constraint
                pos = candidate
                az_accepted = self._azimuth(pos, array_center)
            positions.append(pos)
            azimuths.append(az_accepted)

        return positions,azimuths
    
    def generate_diffuse_noise_numpy(self,input_audio, mic_positions, sr, speed_of_sound=343.0):
        """
        Generate multi-channel diffuse noise from a single-channel source using NumPy.
        
        Args:
            input_audio (np.ndarray): Single-channel audio signal (T,)
            mic_positions (np.ndarray): Microphone coordinates (M, 3) in meters
            sr (int): Sampling rate
            speed_of_sound (float): Speed of sound in m/s
            
        Returns:
            np.ndarray: Multi-channel diffuse noise (M, T)
        """

        # TODO : Add Random to mic position

        # 1. Parameter Settings
        n_fft = 1024
        hop_length = n_fft // 4
        num_mics = mic_positions.shape[0]
        num_samples = len(input_audio)
        
        # 2. STFT and Initial Decorrelation (Random Phase)
        # stft shape: (1 + n_fft//2, n_frames)
        spec = rs.stft(input_audio, n_fft=n_fft, hop_length=hop_length)
        num_bins, num_frames = spec.shape
        
        # Replicate to M channels and add independent random phase [0, 2*pi]
        # This sets initial coherence to 0 across all channels
        random_phase = np.exp(2j * np.pi * np.random.rand(num_mics, num_bins, num_frames))
        spec_m = spec[np.newaxis, :, :] * random_phase # (M, F, T)
        
        # 3. Spatial Coherence Matrix (Sinc)
        # Distance matrix between all mics (M, M)
        dist_matrix = np.linalg.norm(mic_positions[:, np.newaxis, :] - mic_positions[np.newaxis, :, :], axis=2)
        
        # Frequency bins (F,)
        freqs = np.linspace(0, sr / 2, num_bins)
        
        # Output buffer
        output_spec = np.zeros_like(spec_m, dtype=complex)
        
        # 4. Apply Spatial Correlation per Frequency Bin
        for k in range(num_bins):
            if k == 0:
                # DC component: Full correlation (all ones matrix)
                gamma = np.ones((num_mics, num_mics))
            else:
                # Coherence: sinc(2 * pi * f * d / c)
                arg = 2 * np.pi * freqs[k] * dist_matrix / speed_of_sound
                gamma = np.sin(arg) / (arg + 1e-8)  # Add small value to avoid division by zero

                # Randomize Spatial Correlation
                noise_matrix = np.random.normal(0, 0.01, size=(num_mics, num_mics))
                gamma += (noise_matrix + noise_matrix.T) / 2 # Keep postive-definite

                np.fill_diagonal(gamma, 1.0)
                

            # Randommize Sensor noise 
            eps =np.random.uniform(1e-6,1e-4)
            # Regularization for numerical stability
            gamma += np.eye(num_mics) * eps
            
            # Cholesky Decomposition: Gamma = L * L^H
            try:
                L = np.linalg.cholesky(gamma)
            except np.linalg.LinAlgError:
                # Fallback if matrix is not positive-definite
                eig_val, eig_vec = np.linalg.eigh(gamma)
                eig_val = np.maximum(eig_val, 1e-6)
                L = eig_vec @ np.diag(np.sqrt(eig_val))
                
            # Transform independent noise to spatially correlated noise
            # L (M, M) * spec_m (M, T) -> (M, T)
            output_spec[:, k, :] = L @ spec_m[:, k, :]
            
        # 5. Inverse STFT
        diffuse_audio = []
        for m in range(num_mics):
            y_out = rs.istft(output_spec[m], hop_length=hop_length, length=num_samples)
            diffuse_audio.append(y_out)
            
        return np.array(diffuse_audio)

    # --- Usage Example ---
    # mic_pos = np.array([[0, 0, 0], [0.05, 0, 0]]) # 5cm spacing
    # noise_source, _ = librosa.load("noise.wav", sr=16000)
    # multi_noise = self.generate_diffuse_noise_numpy(noise_source, mic_pos, 16000)

    def _simulate_rir(
        self,
        room_dim: np.ndarray,
        rt60: float,
        src_pos: np.ndarray,
        mic_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate RIRs using gpuRIR.

        Parameters
        ----------
        room_dim : (3,)
        rt60     : float
        src_pos  : (3,)  single source position
        mic_pos  : (3, n_mics)

        Returns
        -------
        rirs : (n_mics, rir_len)
        """
        beta   = gpuRIR.beta_SabineEstimation(room_dim, rt60)
        Tmax   = rt60
        nb_img = gpuRIR.t2n(Tmax, room_dim)

        # gpuRIR expects pos_src: (n_src, 3), pos_rcv: (n_rcv, 3)
        pos_src = src_pos[None, :]  # (1, 3)
        pos_rcv = mic_pos.T         # (n_mics, 3)

        # Returns (n_src, n_rcv, rir_len)
        rirs = gpuRIR.simulateRIR(
            room_dim, beta, pos_src, pos_rcv,
            nb_img, Tmax, self.sr,
        )
        return rirs[0]  # (n_mics, rir_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_from_rir(
        self,
        cleans: List[np.ndarray],
        clean_SIRs: List[float],
        rirs_target: np.ndarray,
        rirs_interf: List[np.ndarray],
        mic_pos_rel: np.ndarray,
        noise: Optional[List[np.ndarray]] = None,
        noise_SIRs: List[float] = None,
        SNR: float = None,
        scale: float = 1.0,
    ) -> tuple:
        """
        Mix speech and noise using pre-loaded RIRs (skips gpuRIR simulation).

        Parameters
        ----------
        cleans       : list of (T,) anechoic signals; cleans[0] is the target
        clean_SIRs   : list of SIR values (dB); clean_SIRs[0] is ignored (reference)
        rirs_target  : (n_mics, rir_len) pre-loaded target RIR
        rirs_interf  : list of (n_mics, rir_len) pre-loaded interferer RIRs
        mic_pos_rel  : (n_mics, 3) mic positions relative to array centre
        noise        : list of (T,) noise signals; index 0 is treated as diffuse
        noise_SIRs   : list of SIR values (dB) for noise
        SNR          : SNR in dB (target vs all noise); None = no scaling
        scale        : output amplitude scale

        Returns
        -------
        noisy  : (n_mics, T)
        result : dict with same keys as simulate()
        """
        n_speakers = len(cleans)
        T          = len(cleans[0])
        n_mics     = rirs_target.shape[0]

        def _convolve(sig: np.ndarray, rir: np.ndarray) -> np.ndarray:
            out = fftconvolve(sig, rir)[:T]
            if len(out) < T:
                out = np.pad(out, (0, T - len(out)))
            return out

        def _norm(x: np.ndarray, ret_scalar=False) -> np.ndarray:
            scalar = np.max(np.abs(x)) + 1e-8
            return x / scalar if not ret_scalar else (x / scalar, scalar)

        # ---- target ----
        reverb_target = np.stack(
            [_convolve(cleans[0], rirs_target[m]) for m in range(n_mics)]
        )
        clean_scalar = 1.0
        reverb_target, scalar = _norm(reverb_target, ret_scalar=True)
        clean_scalar /= scalar
        target_rms = _rms(reverb_target[0], vad=True)

        # ---- interferers ----
        reverb_list = []
        for i, rirs_i in enumerate(rirs_interf, start=1):
            reverb_i = np.stack(
                [_convolve(cleans[i], rirs_i[m]) for m in range(n_mics)]
            )
            reverb_i = _norm(reverb_i)
            sir_scale = (target_rms / _rms(reverb_i[0], vad=True)) / (10 ** (clean_SIRs[i] / 20))
            reverb_list.append(reverb_i * sir_scale)

        noisy = reverb_target.copy()
        for reverb_i in reverb_list:
            noisy += reverb_i

        # ---- noise (diffuse only — no gpuRIR for noise sources) ----
        noise_list = []
        noise_ref  = None
        if noise:
            for idx_n, n in enumerate(noise):
                noise_trimmed = n[:T]
                if len(noise_trimmed) < T:
                    noise_trimmed = np.pad(noise_trimmed, (0, T - len(noise_trimmed)))
                noise_mc = self.generate_diffuse_noise_numpy(noise_trimmed, mic_pos_rel, self.sr)
                noise_mc = _norm(noise_mc)
                if noise_mc.shape[1] > T:
                    noise_mc = noise_mc[:, :T]
                elif noise_mc.shape[1] < T:
                    noise_mc = np.pad(noise_mc, ((0, 0), (0, T - noise_mc.shape[1])))
                noise_mc = noise_mc.astype(np.float32)

                if noise_ref is None:
                    noise_ref = noise_mc
                else:
                    noise_scale = (
                        (_rms(noise_ref[0]) / _rms(noise_mc[0]))
                        / (10 ** (noise_SIRs[idx_n] / 20))
                    )
                    noise_mc = noise_mc * noise_scale
                noise_list.append(noise_mc)

        # ---- SNR ----
        noise_rms = 0.0
        for noise_i in noise_list:
            noise_rms += _rms(noise_i[0]) ** 2
        noise_rms = np.sqrt(noise_rms)
        if SNR is not None and noise_rms > 1e-7:
            snr_scale = target_rms / (10 ** (SNR / 20)) / noise_rms
        else:
            snr_scale = 1.0
        for noise_i in noise_list:
            noisy += noise_i * snr_scale

        # ---- final normalisation ----
        norm_denom    = np.abs(noisy).max() + 1e-8
        noisy         = noisy         / norm_denom * scale
        reverb_target = reverb_target / norm_denom * scale
        clean_scalar  = clean_scalar  / norm_denom * scale

        result = {
            'reverb'        : reverb_list,
            'reverb_target' : reverb_target,
            'rir_target'    : rirs_target,
            'mic_pos'       : mic_pos_rel,
            'clean_scalar'  : clean_scalar,
        }
        return noisy, result

    def simulate(
        self,
        cleans: List[np.ndarray],
        clean_SIRs: List[float],
        noise: Optional[np.ndarray] = None,
        noise_SIRs : List[float] = None,
        SNR : float = None,
        scale : float = 1.0,

    ) -> dict:
        """
        Simulate multi-channel room acoustics for one or more speech signals
        and, optionally, a noise signal.

        Parameters
        ----------
        cleans : list of (T,) arrays, length >= 1
            Anechoic speech signal(s).  The first element is the target
            speaker; any additional element is an interfering speaker.
        clean_SIRs : list of SIR values (dB), First speaker as reference
        diffuse noise : (T,)array

        noise_SIRs : list of SIR values (dB) for noise, First noise as reference,
        SNR : SNR value (dB) only first speaker as speech, rest of cleans and whole noise as noise
              -> SNR between S[0] and (S[1:] + N[:])

        Returns
        -------
        dict with keys:
            'reverb'         : list of (n_mics, T)       – one entry per speaker
            'rir_speech'     : list of (n_mics, rir_len) – one entry per speaker
            'noise_mc'       : (n_mics, T)               – present only when noise given
            'mic_pos'        : (n_mics, 3)               – Cartesian, relative to array centre
            'room_params'    : dict  room_dim, rt60, array_center, source_positions

        Steps 
        --------------
        1. Set Mic Array Geometry and Room Parameters
        2. Generate First Speaker(Target)
        3. Generate Interfering Speakers(if needed)
        4. Generate Diffuse Noise(if needed)
        5. Generate Interfering Noise(if needed)
        6. Set SNR
        7. Scaling
        """
        n_speakers = len(cleans)
        T          = len(cleans[0])
        n_mics     = self.cfg.mic_array.n_mics

        room_dim = self._random_room_dim()
        rt60     = np.random.uniform(
            self.cfg.room.rt60_range[0],
            self.cfg.room.rt60_range[1],
        )

        #print(f"SIR : {clean_SIRs} dB, Noise SIR : {noise_SIRs} dB, SNR : {SNR:.2f} dB")

        #################################################
        ### 1. Set Array Geometry and Room parameters ###
        #################################################

        margin = 0.5
        h_mic  = min(float(self.cfg.mic_array.height), room_dim[2] - 0.3)
        array_center = np.array([
            np.random.uniform(margin, room_dim[0] - margin),
            np.random.uniform(margin, room_dim[1] - margin),
            h_mic,
        ])

        mic_pos     = self._mic_positions(array_center)      # (3, n_mics)
        mic_pos_rel = (mic_pos - array_center[:, None]).T    # (n_mics, 3)

        min_angle_diff = float(getattr(self.cfg.source, 'min_angle_diff', 0))
        speech_positions,speech_azimuths = self._sample_speaker_positions(
            n_speakers, room_dim, array_center, min_angle_diff,
        )

        def _convolve(sig: np.ndarray, rir: np.ndarray) -> np.ndarray:
            out = fftconvolve(sig, rir)[:T]
            if len(out) < T:
                out = np.pad(out, (0, T - len(out)))
            return out

        def _norm(x:np.ndarray,ret_scalar=False) -> np.ndarray:
            scalar = np.max(np.abs(x)) + 1e-8
            return x / scalar if not ret_scalar else (x / scalar, scalar)

        #########################################
        ### 2. Generate First Speaker(Target) ###
        #########################################

        rirs_target   = self._simulate_rir(room_dim, rt60, speech_positions[0], mic_pos)
        reverb_target = np.stack(
            [_convolve(cleans[0], rirs_target[m]) for m in range(n_mics)]
        )  # (n_mics, T)

        # norm reverb target
        clean_scalar = 1.0
        reverb_target, scalar = _norm(reverb_target,ret_scalar=True)
        clean_scalar /= scalar
        target_rms = _rms(reverb_target[0],vad=True)


        ####################################################
        ### 3. Generate Interfering Speakers(if needed)  ###
        ####################################################
        #rirs_list   = [rirs_target]
        reverb_list = []

        for i in range(1, n_speakers):
            rirs_i   = self._simulate_rir(room_dim, rt60, speech_positions[i], mic_pos)
            reverb_i = np.stack(
                [_convolve(cleans[i], rirs_i[m]) for m in range(n_mics)]
            )  # (n_mics, T)
            # Scale speaker i to clean_SIRs[i] dB relative to target at mic 0
            reverb_i = _norm(reverb_i)
            sir_scale = (target_rms / _rms(reverb_i[0],vad=True)) / (10 ** (clean_SIRs[i] / 20))
            #rirs_list.append(rirs_i)
            #print(f"Interfering speaker {i}: position={speech_positions[i]}, SIR={clean_SIRs[i]} dB, scale={sir_scale:.3f}| {_rms(reverb_target[0]):.6f} vs {_rms(reverb_i[0]):.6f}")
            reverb_i = reverb_i * sir_scale
            reverb_list.append(reverb_i)

        noisy = reverb_target.copy()
        for i,reverb_i in enumerate(reverb_list):
            noisy += reverb_i
            #sf.write(f"reverb_{i}.wav", reverb_i.T, self.sr)

        ################################################
        ### 4. Generate Diffuse Noise(if needed)     ###
        ################################################

        noise_list = []
        noise_positions=[]
        noise_ref = None
        for idx_n, n in enumerate(noise) :
            # First Noise is always Diffuse Noise
            if idx_n == 0 : 
                noise_trimmed = n[:T]
                if len(noise_trimmed) < T:
                    noise_trimmed = np.pad(noise_trimmed, (0, T - len(noise_trimmed)))
                noise_mc = self.generate_diffuse_noise_numpy(noise_trimmed, mic_pos_rel, self.sr)
                noise_mc = _norm(noise_mc)
                if noise_mc.shape[1] > T:
                    noise_mc = noise_mc[:, :T]
                elif noise_mc.shape[1] < T:
                    noise_mc = np.pad(noise_mc, ((0, 0), (0, T - noise_mc.shape[1])))
                noise_mc = noise_mc.astype(np.float32)

                noise_ref = noise_mc

                noise_list.append(noise_mc)
                noise_positions.append(None)  # No specific position for diffuse noise
                
        ################################################
        ### 5. Generate Interfering Noise(if needed) ###
        ################################################
            # Additional Noise is always Directive Noise
            else :
                noise_pos  = self._random_source_pos(room_dim, array_center)
                rirs_noise = self._simulate_rir(room_dim, rt60, noise_pos, mic_pos)
                noise_mc   = np.stack(
                    [_convolve(n, rirs_noise[m]) for m in range(n_mics)]
                )
                noise_mc = _norm(noise_mc)

                if noise_ref is not None:
                    noise_scale = (_rms(noise_ref[0]) / _rms(noise_mc[0])) / (10 ** (noise_SIRs[idx_n] / 20))
                else :
                    noise_ref = noise_mc
                    noise_scale = 1.0
                noise_mc    = noise_mc * noise_scale
                noise_list.append(noise_mc)
                noise_positions.append(noise_pos)

        ##################
        ### 7. Set SNR ###
        ##################

        noise_rms = 0.0
        #for reverb_i in reverb_list:
        #    noise_rms += _rms(reverb_i[0],vad=True) ** 2
        for noise_i in noise_list : 
            noise_rms += _rms(noise_i[0]) ** 2
        noise_rms = np.sqrt(noise_rms)
        if SNR is not None and noise_rms > 1e-7:
            snr_scale = target_rms/(10 ** (SNR / 20))  / noise_rms
        else :
            snr_scale = 1.0

        for i,noise_i in enumerate(noise_list):
            noisy += noise_i*snr_scale
            #sf.write(f"noise_{i}.wav", noise_i.T, self.sr)

        ##################
        ### 8. Scaling ###
        ##################

        # normalize 
        norm_denom = np.abs(noisy).max() + 1e-8
        noisy = noisy / norm_denom
        reverb_target = reverb_target / norm_denom
        clean_scalar = clean_scalar / norm_denom

        # apply scale
        noisy = noisy * scale
        reverb_target = reverb_target * scale
        clean_scalar = clean_scalar * scale

        result = {
            'reverb'         : reverb_list,      # list of (n_mics, T) — interfering speakers
            'reverb_target'  : reverb_target,    # (n_mics, T) — target speaker, co-normalised with noisy
            'rir_target'     : rirs_target,      # (n_mics, rir_len) — raw target RIR (for early-reflection extraction)
            'mic_pos'        : mic_pos_rel,      # (n_mics, 3) Cartesian, relative
            'target_pos' : speech_positions[0],  # (3,) Cartesian, absolute
            'speech_azimuths': speech_azimuths,  # list of azimuth angles for each speaker
            'room_params'    : {
                'room_dim'        : room_dim,
                'rt60'            : rt60,
                'array_center'    : array_center,
                'source_positions': speech_positions,
                'noise_positions' : noise_positions,
            },
            "clean_scalar" : clean_scalar,       # scalar to recover original clean level from reverb_target
        }

        return noisy,result

def sample_test():
    import librosa as rs
    import soundfile as sf
    from box import Box

    sr = 16000
    n_sample = sr*4

    path_speech_1 = "../samples/male_1.wav"
    path_speech_2 = "../samples/female_1.wav"
    path_noise_2 = "../samples/BGD_150203_010_CAF.CH1_10sec.wav"

    speech_1, _ = rs.load(path_speech_1, sr=sr)
    speech_2, _  = rs.load(path_speech_2, sr=sr)
    noise_1 = np.random.randn(n_sample)  # 10 seconds of white noise
    noise_2, _     = rs.load(path_noise_2, sr=sr)

    # Match Audio Lengths 
    def match_length(signal, n_length):
        if len(signal) < n_length:
            pre_pad = np.random.randint(n_length - len(signal))
            post_pad = n_length - len(signal) - pre_pad
            return np.pad(signal, (pre_pad, post_pad))
        else:
            return signal[:n_length]
        
    speech_1 = match_length(speech_1, n_sample)
    speech_2 = match_length(speech_2, n_sample)
    noise_1   = match_length(noise_1, n_sample)
    noise_2   = match_length(noise_2, n_sample)

    hp = {
        'audio': {
            'sr': sr,
        },
        'sim': {
            'room': {
                'min_dim': [4, 4, 2.5],
                'max_dim': [10, 10, 4],
                'rt60_range': [0.3, 0.8],
            },
            'mic_array': {
                'type': 'linear',
                'n_mics': 4,
                'spacing': 1.00,
                'height': 1.5,
            },
            'source': {
                'height_range': [1.2, 1.8],
                'min_distance': 0.5,
                'max_distance': 4.0,
                'min_angle_diff': 30,
            },
            'noise': {
                'type': 'diffuse',
            },
        },
    }
    hp = Box(hp)
    sim = RIRSimulator(hp)
    noisy, result = sim.simulate(
        cleans=[speech_1],
        clean_SIRs=[0],
        noise=[noise_1],
        noise_SIRs=[0],
        SNR=15,
    )
    print(result)
    sf.write("simulated_noisy.wav", noisy.T, sr)

    
if __name__ == "__main__":
    sample_test()

   