# Dataset

On-the-fly multi-channel speech simulation pipeline for training and evaluating speech enhancement / DOA-estimation models.

---

## Files

| File | Description |
|---|---|
| `DatasetMCSim.py` | `torch.utils.data.Dataset` — synthesises noisy multi-channel mixtures on-the-fly |
| `RIRSimulator.py` | Room-acoustic engine (gpuRIR shoebox model) used by `DatasetMCSim` |
| `Augmentation.py` | Stand-alone audio augmentation helpers (noise generation, biquad EQ, resample, clipping) |

---

## DatasetMCSim

`DatasetMCSim(hp, is_train=True)` — multi-channel simulation dataset.

### How a training sample is built

1. Randomly draw 1–`max_speakers` speech files and 1–`max_noises` noise files.
2. The first speech file is the **target speaker**; additional speech files are **interfering speakers**.
3. Noise index 0 is always **diffuse** (spatially-correlated sinc coherence model); noise index 1+ are **directional point sources** simulated with gpuRIR.
4. `RIRSimulator.simulate()` applies room acoustics, scales interfering speakers to the configured SIR range, and scales noise to the configured SNR.
5. Extract the **clean reference** from the target speaker according to `clean_ref`.
6. Jointly normalise noisy mixture and clean reference to a random target dBFS.

### Sample dict (training)

| Key | Shape | Description |
|---|---|---|
| `noisy` | `(n_mics, T)` | Full mixture (target + interferers + noise) |
| `clean` | `(T,)` | Clean reference for the target speaker at `ref_mic` |
| `mic_pos` | `(n_mics, 3)` | Mic positions relative to array centre (metres) |
| `target_pos` | `(3,)` or `(1, 3)` | Cartesian position of the target speaker (absolute) |

### Sample dict (eval, `is_train=False`)

Eval mode loads pre-generated files from `hp.data.dev.{noisy,clean,meta}`:

| Key | Shape | Description |
|---|---|---|
| `noisy` | `(n_mics, T)` | Pre-mixed noisy file |
| `clean` | `(T,)` | Reference clean file |
| `mic_pos` | `(n_mics, 3)` | From JSON metadata |
| `target_pos` | `(1, 3)` | From JSON metadata |

### Clean reference modes (`clean_ref`)

| Value | Description |
|---|---|
| `anechoic` | Direct-path only — convolve dry speech with the RIR up to its peak sample. Trains for full dereverberation. |
| `early_reflection` | Convolve dry speech with the first `early_reflection_time` seconds of the RIR. |
| `reverberant` | Full reverberant signal at `ref_mic` — trains for noise suppression only. |

### Key config keys (`data` section)

| Key | Default | Description |
|---|---|---|
| `speech` | — | Root path(s) for speech files (`.flac` / `.wav`) |
| `noise` | — | Root path(s) for noise files |
| `sr` | `16000` | Sample rate |
| `len_data` | `48000` | Samples per segment (3 s @ 16 kHz) |
| `n_item` | `100000` | Virtual epoch length (not tied to file count) |
| `max_speakers` | `2` | Maximum number of simultaneous speakers |
| `max_noises` | `2` | Maximum number of noise sources (1 diffuse + N-1 directional) |
| `SIR_clean` | `[-5, 5]` | Interfering-speaker SIR range (dB) relative to target |
| `SIR_noise` | `[-5, 5]` | Directional-noise SIR range (dB) relative to diffuse noise |
| `SNR` | `[-5, 20]` | Target-vs-noise SNR range (dB) |
| `target_dB_FS` | `-25` | Centre level for dBFS normalisation |
| `target_dB_FS_floating_value` | `10` | ±dBFS jitter around `target_dB_FS` |
| `clean_ref` | `anechoic` | Clean reference mode (see above) |
| `ref_mic` | `0` | Reference microphone index (0-based) |
| `rir_dir` | `None` | Optional path to pre-generated RIR cache (skips gpuRIR at runtime) |

### Pre-generated RIR cache (`rir_dir`)

If `hp.data.rir_dir` is set, `DatasetMCSim` loads RIRs from disk instead of calling gpuRIR per sample. The directory must contain:

```
rir_dir/
├── meta/
│   ├── 000001.json   # one JSON per RIR set
│   └── ...
└── *.npy             # RIR arrays referenced in the JSON files
```

Each JSON file must contain:

| Field | Description |
|---|---|
| `n_speakers` | Number of speakers in this set |
| `rir_target` | Relative path to `.npy` file with target RIR `(n_mics, rir_len)` |
| `target_azimuth` | Azimuth angle of the target speaker (degrees) |
| `mic_pos_rel` | `(n_mics, 3)` mic positions relative to array centre |
| `target_pos` | `(3,)` absolute Cartesian position of the target speaker |

---

## RIRSimulator

`RIRSimulator(hp)` — wraps [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) to simulate shoebox rooms.

### `simulate(cleans, clean_SIRs, noise, noise_SIRs, SNR, scale) → (noisy, result)`

Full on-the-fly simulation: samples a random room, mic array, and source positions, then renders all signals.

**Returns**

| | Shape | Description |
|---|---|---|
| `noisy` | `(n_mics, T)` | Fully mixed and normalised signal |
| `result['reverb_target']` | `(n_mics, T)` | Reverberant target speaker, co-normalised with `noisy` |
| `result['rir_target']` | `(n_mics, rir_len)` | Raw RIR for the target speaker |
| `result['reverb']` | `list[(n_mics, T)]` | Reverberant interfering speakers |
| `result['mic_pos']` | `(n_mics, 3)` | Mic positions relative to array centre |
| `result['target_pos']` | `(3,)` | Absolute Cartesian position of target speaker |
| `result['speech_azimuths']` | `list[float]` | Azimuth angles of all speakers (degrees) |
| `result['clean_scalar']` | `float` | Scale factor: multiply reverb_target by this to recover original level |
| `result['room_params']` | `dict` | `room_dim`, `rt60`, `array_center`, `source_positions`, `noise_positions` |

### `simulate_from_rir(cleans, clean_SIRs, rirs_target, rirs_interf, mic_pos_rel, noise, noise_SIRs, SNR) → (noisy, result)`

Same mixing logic as `simulate()` but loads pre-computed RIRs — skips the gpuRIR call. Used by `DatasetMCSim._get_sample_from_rir()`. All noise sources are treated as diffuse (no gpuRIR for noise).

### Mic array geometries (`sim.mic_array.type`)

| Type | Description | Relevant keys |
|---|---|---|
| `linear` | Uniform Linear Array (ULA) along the X axis | `n_mics`, `spacing` |
| `circular` | Uniform Circular Array (UCA) in the XY plane | `n_mics`, `radius` |
| `dynamic` | Random positions with pairwise distance constraints; falls back to circular if no valid layout found in 1000 trials | `n_mics`, `min_pair_dist`, `max_pair_dist` |

### Key config keys (`sim` section)

```yaml
sim:
  room:
    min_dim: [3.0, 3.0, 2.5]   # metres [x, y, z]
    max_dim: [10.0, 10.0, 4.0]
    rt60_range: [0.1, 0.8]     # seconds

  mic_array:
    type: linear                # 'linear' | 'circular' | 'dynamic'
    n_mics: 4
    spacing: 0.05               # metres (linear only)
    radius: 0.05                # metres (circular only)
    min_pair_dist: 0.05         # metres (dynamic only)
    max_pair_dist: 0.30         # metres (dynamic only)
    height: 1.2                 # array centre height from floor (metres)

  source:
    min_distance: 0.5           # metres from array centre
    max_distance: 5.0
    height_range: [1.0, 2.0]   # metres
    min_angle_diff: 10          # degrees — minimum azimuth separation between speakers
```

### Diffuse noise model

`generate_diffuse_noise_numpy(input_audio, mic_positions, sr)` generates spatially correlated multi-channel noise from a mono source using the sinc coherence model:

- Per frequency bin *k*: coherence matrix `Γ[i,j] = sinc(2πf_k d_ij / c)`
- Cholesky decomposition of `Γ` colours independent per-channel noise to the correct spatial correlation.
- Falls back to eigen-decomposition if the matrix is not positive-definite.

---

## Augmentation

Stand-alone helpers in `Augmentation.py`. None are currently called from `DatasetMCSim`; apply them externally before passing audio to the simulator if needed.

### `gen_noise(n_sample, n_freq, f_decay, cutoff, sr) → np.ndarray`

Generates coloured noise by shaping white noise in the frequency domain.

| `f_decay` | Noise colour |
|---|---|
| `0.0` | White |
| `1.0` | Pink |
| `2.0` | Brown |
| `-1.0` | Blue |

> **Note:** Pink/Brown noise generation has a known bug (`f_decay > 0.0`).

### `rand_biquad_filter(x, sr, max_iter, gain_db_high, gain_db_low, q_low, q_high) → np.ndarray`

Applies 1–`max_iter` random biquad filters in series. Filter types drawn uniformly:
`high_shelf`, `high_pass`, `low_shelf`, `low_pass`, `peaking_eq`, `notch`.

Individual filter design functions are also exported:
`high_shelf`, `high_pass`, `low_shelf`, `low_pass`, `peaking_eq`, `notch` — each returns `(b, a)` SOS coefficients for use with `scipy.signal.lfilter`.

### `rand_resample(x, sr, r_low, r_high) → np.ndarray`

Resamples `x` to a randomly perturbed sample rate in `[r_low·sr, r_high·sr]`, rounded to the nearest 500 Hz, then crops back to the original length.

### `rand_clipping(x, c_min, c_max) → np.ndarray`

Hard-clips `x` to a random threshold in `[c_min·max|x|, c_max·max|x|]`.

### `remove_dc(x) → np.ndarray`

Subtracts the mean along the last axis (DC removal).

---

## Quick smoke-test

```bash
# DatasetMCSim
conda run -n dnn python Dataset/DatasetMCSim.py

# RIRSimulator standalone
conda run -n dnn python Dataset/RIRSimulator.py
```

`RIRSimulator.__main__` writes `simulated_noisy.wav` to the working directory using the inline `sample_test()` function.
