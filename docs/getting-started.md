# Getting Started

Get audiolib installed and processing audio in under five minutes.

---

## Installation

```bash
pip install audiolib
```

Prebuilt wheels are available for:
- Linux x86_64 and aarch64
- macOS Intel (x86_64) and Apple Silicon (arm64)
- Windows x86_64

Python 3.8 through 3.13 are supported. No Rust installation required.

### Optional extras

For higher-quality resampling, install [soxr](https://github.com/dofuuz/python-soxr):
```bash
pip install soxr
```
audiolib will automatically use it when available, falling back to the built-in Rust resampler otherwise.

For effects that use median filtering (HPSS), install scipy:
```bash
pip install scipy
```

---

## Building from source

If you want to contribute or need a custom build, you'll need the Rust toolchain.

```bash
# 1. Clone the repository
git clone https://github.com/LimaBD/audiolib
cd audiolib

# 2. Install everything and build in dev mode
./scripts/dev_install.sh
```

The script installs Rust (via rustup if needed), maturin, and all dev dependencies, then runs `maturin develop --release` to compile the Rust extension and install the package in editable mode.

Manual steps if you prefer:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install "maturin>=1.5,<2.0" numpy soundfile pytest
maturin develop --release
```

---

## Verify the installation

```python
import audiolib
print(audiolib.__version__)   # 0.1.0
```

---

## Your first pipeline

### Load audio

```python
import audiolib

# Load at native sample rate
y, sr = audiolib.load("audio.wav", sr=None)

# Load and resample to 22 kHz mono
y, sr = audiolib.load("audio.wav", sr=22050, mono=True)

# Load a specific segment (seconds)
y, sr = audiolib.load("audio.wav", offset=10.0, duration=5.0)

print(f"{len(y)/sr:.2f}s of audio @ {sr} Hz")
```

### Compute a spectrogram

```python
import audiolib
import numpy as np

y, sr = audiolib.load("audio.wav", sr=22050)

# Short-time Fourier transform → complex (n_bins, n_frames)
D = audiolib.stft(y, n_fft=2048, hop_length=512)

# Magnitude in decibels
D_db = audiolib.amplitude_to_db(np.abs(D))
print(D_db.shape)   # (1025, n_frames)
```

### Extract mel-frequency features

```python
import audiolib

y, sr = audiolib.load("audio.wav", sr=22050)

# Mel spectrogram — power in mel-frequency bands
M = audiolib.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Log-compressed for ML input
M_db = audiolib.power_to_db(M)

# MFCCs — 13 coefficients per frame
mfcc = audiolib.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print(mfcc.shape)   # (13, n_frames)
```

### Chroma and spectral features

```python
import audiolib

y, sr = audiolib.load("audio.wav", sr=22050)

# 12-bin chromagram — useful for harmony and chord analysis
chroma = audiolib.feature.chroma_stft(y=y, sr=sr)

# Spectral centroid — "brightness" over time
centroid = audiolib.feature.spectral_centroid(y=y, sr=sr)

# Onset strength envelope
oenv = audiolib.feature.onset_strength(y=y, sr=sr)

print(chroma.shape)   # (12, n_frames)
```

### Apply effects

```python
import audiolib

y, sr = audiolib.load("audio.wav", sr=22050)

# Shift pitch up by 4 semitones
y_pitched = audiolib.effects.pitch_shift(y, sr=sr, n_steps=4)

# Slow down to 75% speed
y_slow = audiolib.effects.time_stretch(y, rate=0.75)

# Separate harmonic and percussive components
harmonic, percussive = audiolib.effects.hpss(y)

# Remove silence at start and end
y_trimmed, index = audiolib.effects.trim(y, top_db=60)
```

### Unit conversions

```python
import audiolib
import numpy as np

# Single values or full arrays — same function
audiolib.hz_to_mel(440.0)                         # 69.86
audiolib.hz_to_mel(np.array([110., 220., 440.]))  # array

audiolib.hz_to_note(440.0)    # "A4"
audiolib.note_to_midi("C4")  # 60

# Frame indices ↔ time
audiolib.frames_to_time(np.arange(10), hop_length=512, sr=22050)
```

---

## Processing many files

```python
import audiolib
from pathlib import Path
import numpy as np

def extract_features(path: str) -> np.ndarray:
    y, sr = audiolib.load(path, sr=22050, mono=True)
    mfcc = audiolib.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)  # mean over time → (13,) vector

audio_files = list(Path("dataset/").glob("**/*.wav"))
features = np.stack([extract_features(str(p)) for p in audio_files])
print(features.shape)  # (n_files, 13)
```

Because audiolib functions release the GIL during Rust execution, you can combine this pattern with `concurrent.futures.ThreadPoolExecutor` to saturate multiple CPU cores.

---

## Running the tests

```bash
# Full test suite
pytest tests/ -v

# Skip the benchmark file
pytest tests/ -v --ignore=tests/benchmark.py

# Cross-check against librosa
pip install librosa
LIBROSA_COMPAT=1 pytest tests/test_compat.py -v
```

---

## Next steps

- [API reference](api-reference.md) — complete function signatures and parameters
- [Performance guide](performance.md) — benchmarks and tips for maximum throughput
