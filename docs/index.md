# audiolib — Audio processing at the speed of Rust

> Process audio very fast. Build production ML pipelines without waiting for Python.

---

## What is audiolib?

audiolib is a Python library for audio analysis and digital signal processing. It covers the full stack: loading audio files, computing spectrograms, extracting features for machine learning, applying effects, and converting between units.

What makes it different is what happens when you call a function. Instead of executing pure Python or delegating to a Python-level NumPy loop, audiolib dispatches your work to a compiled Rust extension — [rustfft](https://github.com/ejmahler/RustFFT), hand-tuned FFT butterflies, cache-friendly memory layouts. The kind of performance that usually requires dropping into C or writing CUDA kernels.

You get none of that complexity. You write Python. The Rust engine runs underneath.

---

## Who is it for?

**ML engineers** building audio feature extraction pipelines that need to process thousands of files without the compute becoming a bottleneck.

**Researchers** who prototype in Python and want production performance without rewriting anything.

**Backend developers** embedding real-time audio analysis in services where latency budgets are tight.

**Anyone** who has ever stared at a progress bar wondering why computing a mel spectrogram takes so long.

---

## Core ideas

### Native speed, Python ergonomics

The heavy operations — Short-Time Fourier Transforms, mel filter bank convolutions, DCT for MFCCs, spectral centroid calculations — are all implemented in Rust and compiled to native machine code. Python only touches the surface: argument parsing, NumPy array I/O, dispatch.

### Works at any scale

Whether you're analyzing a single 3-second clip or batching through 500,000 audio files overnight, the performance characteristics stay consistent. No GIL contention in the hot path, no Python-level loops over individual samples.

### Familiar API design

audiolib follows the same conventions and parameter names used by the wider Python audio ecosystem. If you know how `stft(y, n_fft=2048, hop_length=512)` works, you already know how to use audiolib.

### Zero-dependency installation

Prebuilt binary wheels are published to PyPI for every supported platform and Python version. `pip install audiolib` pulls a self-contained package — no Rust compiler, no system libraries, no setup beyond pip.

---

## Feature overview

| Capability | Details |
|---|---|
| **Audio I/O** | Load/save via soundfile, automatic mono conversion, resampling |
| **Spectral transforms** | STFT, ISTFT (reconstruction), magnitude/phase decomposition |
| **Mel features** | Mel spectrogram (Slaney & HTK scales), MFCCs, mel filter banks |
| **Chroma features** | 12-bin chromagram from STFT |
| **Spectral features** | Centroid, bandwidth, rolloff, flatness, RMS, onset strength |
| **Time-domain** | Zero crossings, autocorrelation, μ-law companding |
| **Effects** | Time stretching, pitch shifting, harmonic/percussive separation (HPSS) |
| **Unit conversions** | Hz ↔ mel ↔ MIDI ↔ note names; frames ↔ samples ↔ time |
| **Utilities** | Framing, padding, normalization, masking |

---

## Next steps

- [Getting started](getting-started.md) — install, build, and run your first pipeline
- [API reference](api-reference.md) — every function, every parameter
- [Performance guide](performance.md) — benchmarks, tuning tips, and when Rust matters most
