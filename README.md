# Pytorch-template

A PyTorch training template for speech enhancement, built around the DNS (Deep Noise Suppression) challenge setup. Provides an on-the-fly audio mixing pipeline with configurable augmentation, loss, and scheduler options.

---

## Installation

```bash
pip install -r requirments.txt
```

---

## Quick Start

### 1. Implement your model

Edit `models/model.py` to define your architecture:

```python
class Model(nn.Module):
    def __init__(self, hp):
        super().__init__()
        # define layers

    def forward(self, noisy):
        # input:  (B, T) waveform
        # output: (B, T) enhanced waveform
        return enhanced
```

Then wire it into `common.py`:

```python
def get_model(hp):
    from models.model import Model
    return Model(hp)
```

### 2. Set up your config

Copy `config/default.yaml` to `config/v0.yaml` and fill in your data paths:

```yaml
data:
  clean: "/path/to/clean/wavs"
  noise: "/path/to/noise/wavs"
  eval:
    clean: "/path/to/eval/clean"
    noisy: "/path/to/eval/noisy"

log:
  root: "/path/to/logs"
```

### 3. Train

```bash
# Using the shell launcher (edit VERSION and DEVICE inside first)
bash train_d0.sh

# Or directly
python train.py -c config/v0.yaml --default config/default.yaml -v v0 -d cuda:0
```

Resume from a checkpoint:

```bash
python train.py -c config/v0.yaml --default config/default.yaml -v v0 -d cuda:0 \
    --chkpt /path/to/logs/chkpt/v0/bestmodel.pt -s <starting_step>
```

### 4. Inference

```bash
python inference.py -c config/v0.yaml -m /path/to/logs/chkpt/v0/bestmodel.pt -o output/
```

---

## Configuration

All options live in YAML files. Values in `-c <config>` override those in `--default <default>`.

| Section | Key options |
|---|---|
| `train` | `epoch`, `batch_size`, `num_workers`, `adam` (lr) |
| `loss.type` | `MSELoss`, `wSDRLoss` |
| `scheduler.type` | `Plateau`, `oneCycle`, `CosineAnnealingLR` |
| `data` | `clean`, `noise`, `RIR`, `SNR`, `sr`, augmentation flags |
| `log` | `root` (output dir), `eval` (metrics: `PESQ_WB`, `SISDR`, `DNSMOS`) |

### Data augmentation flags (under `data:`)

| Flag | Description |
|---|---|
| `use_RIR` | Apply room impulse response (reverberation) |
| `spec_augmentation` | SpecAugment (time/freq masking) |
| `RandomNoise` | Add colored noise (controlled by `f_decay`) |
| `biquad_filter` | Random biquad filter (high/low shelf, peaking, notch, etc.) |
| `rand_resample` | Random resampling within `[r_low, r_high]` |
| `remove_dc` | DC offset removal |

---

## Output Structure

```
<log.root>/
├── chkpt/<version>/
│   ├── bestmodel.pt     # lowest validation loss
│   └── lastmodel.pt     # most recent epoch
└── log/<version>/       # TensorBoard logs
```

Monitor training:

```bash
tensorboard --logdir <log.root>/log
```

---

## Project Layout

```
train.py          # training loop
inference.py      # batch inference
common.py         # get_model(), run(), evaluate()
models/model.py   # model definition (implement here)
data/
  DatasetDNS.py   # on-the-fly mixing dataset
  DatasetVD.py    # pre-paired dataset (VoiceBank+DEMAND)
  Mixer.py        # SNR / RIR / dBFS mixing logic
  Augmentation.py # noise, filters, resampling
config/
  default.yaml    # base config
src/utils/        # hparams, writer, loss, metrics (git submodule)
```
