# Performance Guide

How audiolib achieves its speed, how to measure it, and how to get the most out of it in your pipeline.

---

## Why Rust?

Python is interpreted. Every loop, every arithmetic operation, every function call carries overhead. NumPy mitigates this by delegating array operations to BLAS/LAPACK routines, but complex DSP pipelines — especially those involving FFTs, mel filter bank convolutions, spectral aggregations, and unit conversions over large arrays — still spend significant time in Python dispatch code.

audiolib pushes the entire computation into Rust:

- **No Python-level loops.** STFT computation, mel filter bank projection, DCT for MFCCs — all implemented as tight Rust loops over contiguous f32 memory.
- **rustfft for FFTs.** [rustfft](https://github.com/ejmahler/RustFFT) is a state-of-the-art pure-Rust FFT library that adapts its algorithm to the input size. It consistently outperforms scipy's FFT for the frame sizes typical in audio (512–4096 samples).
- **Cache-friendly layouts.** Rust code works directly on the NumPy buffer's memory without copying to intermediate Python objects. Array access patterns are designed for L1/L2 cache locality.
- **GIL release.** Rust functions release the CPython GIL during execution, meaning CPU-bound audio processing can overlap across threads.

---

## Benchmark: what to expect

The following table shows typical throughput on a 5-second stereo recording at 22 kHz on an AMD Ryzen 7 (Linux x86_64), comparing audiolib against a pure-Python implementation.

| Operation | audiolib | Pure Python / NumPy | Speedup |
|---|---|---|---|
| `stft` (n_fft=2048, hop=512) | ~4 ms | ~18 ms | ~4–5× |
| `feature.melspectrogram` | ~7 ms | ~35 ms | ~4–5× |
| `feature.mfcc` (n_mfcc=13) | ~9 ms | ~45 ms | ~4–5× |
| `feature.chroma_stft` | ~8 ms | ~40 ms | ~4–5× |
| `hz_to_mel` (100k elements) | ~0.2 ms | ~1.5 ms | ~7× |
| `frames_to_time` (10k frames) | ~0.05 ms | ~0.4 ms | ~8× |

> Numbers are illustrative. Run `python tests/benchmark.py` on your own hardware for accurate results.

---

## Running benchmarks

```bash
# After installing audiolib
python tests/benchmark.py

# Side-by-side with librosa
pip install librosa
python tests/benchmark.py   # automatically detects and times librosa when installed
```

The benchmark script (`tests/benchmark.py`) processes 5 seconds of synthetic audio and reports mean latency per operation. It also exposes pytest-benchmark wrappers:

```bash
pip install pytest-benchmark
pytest tests/benchmark.py --benchmark-only --benchmark-sort=mean
```

---

## Tuning tips

### 1. Match `hop_length` to your use case

Smaller `hop_length` → more frames → more compute per second of audio. If you don't need high temporal resolution, increase `hop_length`:

```python
# High temporal resolution (expensive)
M = audiolib.feature.melspectrogram(y=y, sr=sr, hop_length=256)

# Lower resolution, much faster — often sufficient for classification
M = audiolib.feature.melspectrogram(y=y, sr=sr, hop_length=1024)
```

### 2. Reuse the spectrogram

When you need multiple features from the same signal, compute the STFT or mel spectrogram once and pass it as `S`:

```python
import audiolib

y, sr = audiolib.load("audio.wav")

# Compute spectrogram once
S = audiolib.feature.melspectrogram(y=y, sr=sr)

# All of these reuse S — no redundant FFT
mfcc = audiolib.feature.mfcc(S=S, sr=sr)
```

For spectral features, pass the magnitude STFT directly:

```python
D = audiolib.stft(y)
mag = np.abs(D)

centroid  = audiolib.feature.spectral_centroid(S=mag, sr=sr)
bandwidth = audiolib.feature.spectral_bandwidth(S=mag, sr=sr)
rolloff   = audiolib.feature.spectral_rolloff(S=mag, sr=sr)
```

### 3. Parallel file processing with threads

Because audiolib releases the GIL during Rust execution, Python threads actually parallelize on CPU-bound audio work:

```python
from concurrent.futures import ThreadPoolExecutor
import audiolib

def process(path):
    y, sr = audiolib.load(path, sr=22050)
    return audiolib.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

paths = [...]  # list of audio files
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(process, paths))
```

### 4. Use float32 throughout

audiolib internally works with `float32`. If your pipeline loads audio as `float64` (e.g., from another library), convert early:

```python
y = y.astype(np.float32)
```

This halves memory bandwidth and doubles SIMD parallelism in Rust.

### 5. Pre-load at target sample rate

Resampling inside `load()` is convenient but adds latency. If you're processing a large dataset, resample once at the file level and cache the resampled versions. For on-the-fly processing, lower sample rates (e.g., 16 kHz for speech) are substantially cheaper than the default 22 kHz.

```python
y, sr = audiolib.load("audio.wav", sr=16000)  # 16 kHz is plenty for speech
```

### 6. Choose the right resampling quality

`res_type` controls the quality/speed trade-off:

| `res_type` | Quality | Speed |
|---|---|---|
| `"soxr_vhq"` | Highest | Slowest |
| `"soxr_hq"` | High (default) | Fast |
| `"soxr_mq"` | Medium | Faster |
| `"soxr_lq"` | Low | Fastest |
| `"linear"` | Basic (Rust built-in) | Instant |

For offline batch processing, `"soxr_hq"` is the right default. For real-time pipelines with low-latency requirements, `"linear"` or `"soxr_lq"` may be acceptable.

---

## Memory usage

audiolib operates zero-copy wherever possible. The STFT result is materialized as a new NumPy array; the input signal buffer is not modified. Keeping the input buffer alive while processing is safe.

For very long recordings, consider chunked processing:

```python
import audiolib
import numpy as np
import soundfile as sf

CHUNK = 22050 * 10  # 10-second chunks

features = []
with sf.SoundFile("long_recording.wav") as f:
    while True:
        chunk = f.read(CHUNK, dtype="float32", always_2d=False)
        if len(chunk) == 0:
            break
        M = audiolib.feature.melspectrogram(y=chunk, sr=f.samplerate)
        features.append(M)

full = np.concatenate(features, axis=1)
```

---

## Profiling your pipeline

Use Python's built-in `cProfile` or `line_profiler` to find bottlenecks. If the hotspot is inside a audiolib function, the Rust layer is already the fastest it can be — focus on the algorithmic changes described above (fewer frames, reuse spectrograms, parallel threads).

```bash
python -m cProfile -s cumtime your_script.py | head -30
```
