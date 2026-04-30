"""
Performance benchmarks comparing audiolib vs librosa (when installed).

Run with:
    python tests/benchmark.py

Or via pytest-benchmark:
    pytest tests/benchmark.py --benchmark-only
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

SR = 22050
DURATION = 5.0  # seconds — longer signal to get meaningful timing
N_FFT = 2048
HOP_LENGTH = 512

rng = np.random.default_rng(42)
Y = rng.standard_normal(int(SR * DURATION)).astype(np.float32)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timeit(fn: Callable, n_repeats: int = 10) -> tuple:
    """Return (mean_ms, min_ms, max_ms) over n_repeats calls."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.min(times)), float(np.max(times))


def print_row(label: str, lx_ms: float, lr_ms: float | None) -> None:
    if lr_ms is not None:
        speedup = lr_ms / lx_ms
        print(f"  {label:<35}  audiolib: {lx_ms:7.2f}ms   librosa: {lr_ms:7.2f}ms   speedup: {speedup:.2f}x")
    else:
        print(f"  {label:<35}  audiolib: {lx_ms:7.2f}ms   librosa: N/A")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_stft():
    import audiolib.core as lx
    def lx_fn():
        return lx.stft(Y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    lx_ms, _, _ = timeit(lx_fn)

    try:
        import librosa
        def lr_fn():
            return librosa.stft(Y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        lr_ms, _, _ = timeit(lr_fn)
    except ImportError:
        lr_ms = None

    print_row("stft", lx_ms, lr_ms)
    return lx_ms


def bench_istft():
    import audiolib.core as lx
    D = lx.stft(Y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    def lx_fn():
        return lx.istft(D, hop_length=HOP_LENGTH, length=len(Y))
    lx_ms, _, _ = timeit(lx_fn)

    try:
        import librosa
        D_lr = librosa.stft(Y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        def lr_fn():
            return librosa.istft(D_lr, hop_length=HOP_LENGTH, length=len(Y))
        lr_ms, _, _ = timeit(lr_fn)
    except ImportError:
        lr_ms = None

    print_row("istft", lx_ms, lr_ms)


def bench_melspectrogram():
    import audiolib.feature as lxf
    def lx_fn():
        return lxf.melspectrogram(y=Y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    lx_ms, _, _ = timeit(lx_fn)

    try:
        import librosa
        def lr_fn():
            return librosa.feature.melspectrogram(y=Y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
        lr_ms, _, _ = timeit(lr_fn)
    except ImportError:
        lr_ms = None

    print_row("feature.melspectrogram", lx_ms, lr_ms)


def bench_mfcc():
    import audiolib.feature as lxf
    def lx_fn():
        return lxf.mfcc(y=Y, sr=SR, n_mfcc=13)
    lx_ms, _, _ = timeit(lx_fn)

    try:
        import librosa
        def lr_fn():
            return librosa.feature.mfcc(y=Y, sr=SR, n_mfcc=13)
        lr_ms, _, _ = timeit(lr_fn)
    except ImportError:
        lr_ms = None

    print_row("feature.mfcc", lx_ms, lr_ms)


def bench_chroma():
    import audiolib.feature as lxf
    def lx_fn():
        return lxf.chroma_stft(y=Y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    lx_ms, _, _ = timeit(lx_fn)

    try:
        import librosa
        def lr_fn():
            return librosa.feature.chroma_stft(y=Y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
        lr_ms, _, _ = timeit(lr_fn)
    except ImportError:
        lr_ms = None

    print_row("feature.chroma_stft", lx_ms, lr_ms)


def bench_hz_to_mel():
    import audiolib.convert as lxc
    hz = np.linspace(0, 8000, 100_000).astype(np.float32)
    def lx_fn():
        return lxc.hz_to_mel(hz)
    lx_ms, _, _ = timeit(lx_fn, n_repeats=50)

    try:
        import librosa
        hz64 = hz.astype(np.float64)
        def lr_fn():
            return librosa.hz_to_mel(hz64)
        lr_ms, _, _ = timeit(lr_fn, n_repeats=50)
    except ImportError:
        lr_ms = None

    print_row("hz_to_mel (100k elements)", lx_ms, lr_ms)


def bench_onset_strength():
    import audiolib.feature as lxf
    def lx_fn():
        return lxf.onset_strength(y=Y, sr=SR, hop_length=HOP_LENGTH)
    lx_ms, _, _ = timeit(lx_fn)

    try:
        import librosa
        def lr_fn():
            return librosa.onset.onset_strength(y=Y, sr=SR, hop_length=HOP_LENGTH)
        lr_ms, _, _ = timeit(lr_fn)
    except ImportError:
        lr_ms = None

    print_row("feature.onset_strength", lx_ms, lr_ms)


# ---------------------------------------------------------------------------
# pytest-benchmark wrappers (if running under pytest)
# ---------------------------------------------------------------------------

def test_bench_stft(benchmark):
    import audiolib.core as lx
    benchmark(lambda: lx.stft(Y, n_fft=N_FFT, hop_length=HOP_LENGTH))


def test_bench_melspectrogram(benchmark):
    import audiolib.feature as lxf
    benchmark(lambda: lxf.melspectrogram(y=Y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH))


def test_bench_mfcc(benchmark):
    import audiolib.feature as lxf
    benchmark(lambda: lxf.mfcc(y=Y, sr=SR, n_mfcc=13))


def test_bench_hz_to_mel(benchmark):
    import audiolib.convert as lxc
    hz = np.linspace(0, 8000, 100_000).astype(np.float32)
    benchmark(lambda: lxc.hz_to_mel(hz))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\naudiolib benchmark — {DURATION}s of audio @ {SR} Hz")
    print(f"n_fft={N_FFT}, hop_length={HOP_LENGTH}")
    print("-" * 80)
    bench_stft()
    bench_istft()
    bench_melspectrogram()
    bench_mfcc()
    bench_chroma()
    bench_onset_strength()
    bench_hz_to_mel()
    print("-" * 80)
    print("Done.\n")
