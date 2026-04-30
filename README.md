# audiolib ⚡

[![PyPI](https://img.shields.io/pypi/v/audiolib)](https://pypi.org/project/audiolib/)
[![CI](https://github.com/LimaBD/audiolib/actions/workflows/ci.yml/badge.svg)](https://github.com/LimaBD/audiolib/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/audiolib)](https://pypi.org/project/audiolib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Audio processing that keeps up with you.** audiolib runs your DSP pipelines at native speed with a Rust engine under the hood — so you can focus on what you build, not on waiting for it.

**audiolib** brings production-grade audio analysis to Python without sacrificing the developer experience you already know. STFT, mel spectrograms, MFCCs, chroma features, spectral analysis — all compiled to native machine code via Rust, all accessible through a clean, familiar Python API.

```python
import audiolib

y, sr = audiolib.load("track.wav")

# Mel spectrogram in one line — computed in native code
M = audiolib.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# MFCCs for your ML pipeline
mfcc = audiolib.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Frequency conversions at scale — vectorized in Rust
notes = audiolib.hz_to_note([220.0, 440.0, 880.0])  # ["A3", "A4", "A5"]
```

---

## Why audiolib?

### Built for the real world

Python is great for research. But when you're processing thousands of audio files in an ML pipeline, running real-time inference, or shipping a product, every millisecond matters. audiolib was built with that reality in mind.

### Rust at the core, Python at the surface

The heavy lifting — FFTs, filter banks, spectral math, unit conversions — is compiled to native machine code using [Rust](https://www.rust-lang.org/) and exposed to Python via [PyO3](https://pyo3.rs/). You write Python. Rust does the work.

### No friction to adopt

audiolib speaks the same language as the ecosystem. If your existing code already uses audio analysis idioms, you'll be up and running in minutes. No new paradigms. No rewrites. Just faster results.

### Ships as a precompiled wheel

No Rust compiler required for your team or CI. Prebuilt binaries for Linux, macOS (Intel + Apple Silicon), and Windows are published on PyPI for every Python version from 3.8 to 3.13.

---

## What's inside

| Module | What it does |
|---|---|
| `audiolib.core` | Audio I/O, STFT/ISTFT, magnitude scaling, zero crossings, μ-law companding, signal generators |
| `audiolib.feature` | Mel spectrograms, MFCCs, chroma, spectral centroid/bandwidth/rolloff/flatness, RMS, onset strength |
| `audiolib.convert` | Hz ↔ mel ↔ MIDI ↔ note name; frames ↔ samples ↔ time — all vectorized |
| `audiolib.effects` | Time stretching, pitch shifting, harmonic/percussive separation, silence trimming |
| `audiolib.util` | Framing, padding, normalization, masking and other signal utilities |

---

## Installation

```bash
pip install audiolib
```

That's it. Precompiled wheels for all major platforms, no system dependencies.

### Build from source (for contributors)

```bash
git clone https://github.com/LimaBD/audiolib
cd audiolib
./scripts/dev_install.sh   # installs Rust + maturin + dev deps, then maturin develop
```

---

## Quick start

```python
import audiolib
import numpy as np

# Load and normalize
y, sr = audiolib.load("audio.wav", sr=22050)

# Short-time Fourier transform
D = audiolib.stft(y, n_fft=2048, hop_length=512)

# Mel spectrogram → log scale
M = audiolib.feature.melspectrogram(y=y, sr=sr, n_mels=128)
M_db = audiolib.power_to_db(M)

# Full MFCC pipeline
mfcc = audiolib.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Chroma features
chroma = audiolib.feature.chroma_stft(y=y, sr=sr)

# Unit conversions — scalar or array, same call
audiolib.hz_to_mel(440.0)           # scalar
audiolib.hz_to_mel(np.array([220., 440., 880.]))  # vectorized

# Pitch shift by 2 semitones
y_shifted = audiolib.effects.pitch_shift(y, sr=sr, n_steps=2)

# Harmonic / percussive separation
H, P = audiolib.effects.hpss(y)
```

---

## Development

```bash
# Build + install in editable mode
./scripts/dev_install.sh

# Run the full test suite
./scripts/run_tests.sh

# Run performance benchmarks (also compares against librosa when installed)
python tests/benchmark.py

# Cross-check numerical output against librosa
pip install librosa
LIBROSA_COMPAT=1 pytest tests/test_compat.py -v

# Build release wheels locally
./scripts/build.sh
```

---

## Architecture

```
audiolib/
├── native/               # Rust crate (PyO3 extension → audiolib._core)
│   └── src/
│       ├── lib.rs        # Module entry point
│       ├── dsp.rs        # STFT, mel filters, MFCCs, spectral features, YIN
│       └── convert.rs    # Frequency / unit conversions (fully vectorized)
└── src/audiolib/         # Python API layer (numpy ↔ Rust bridge)
    ├── __init__.py       # Public surface
    ├── core.py           # Audio I/O, STFT wrappers, signal generators
    ├── feature.py        # Feature extraction
    ├── convert.py        # Unit conversion wrappers
    ├── effects.py        # Time-stretch, pitch-shift, HPSS
    └── util.py           # Utility helpers
```

The Python layer handles NumPy array marshalling and keyword-argument ergonomics. Rust does every multiply, every FFT butterfly, every filter tap. The boundary between them is invisible to callers.

---

## Docs

Full documentation lives in the [`docs/`](docs/) directory:

- [Overview & motivation](docs/index.md)
- [Getting started](docs/getting-started.md)
- [API reference](docs/api-reference.md)
- [Performance guide](docs/performance.md)

---

## License

MIT — see [LICENSE](LICENSE).
