"""
Shared test fixtures for audiolib test suite.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SR = 22050          # standard sample rate used across tests
DURATION = 1.0      # seconds of synthetic audio
N_SAMPLES = int(SR * DURATION)
N_FFT = 2048
HOP_LENGTH = 512
FREQ_440 = 440.0    # A4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sr() -> int:
    """Default sample rate."""
    return SR


@pytest.fixture(scope="session")
def sine_440(sr) -> np.ndarray:
    """1-second 440 Hz sine wave at the default sample rate."""
    t = np.linspace(0, DURATION, int(sr * DURATION), endpoint=False)
    return np.sin(2.0 * np.pi * FREQ_440 * t).astype(np.float32)


@pytest.fixture(scope="session")
def stereo_sine(sr) -> np.ndarray:
    """2-channel (stereo) 440 Hz sine wave, shape (2, n_samples)."""
    t = np.linspace(0, DURATION, int(sr * DURATION), endpoint=False)
    mono = np.sin(2.0 * np.pi * FREQ_440 * t).astype(np.float32)
    return np.stack([mono, mono * 0.5])


@pytest.fixture(scope="session")
def chirp_signal(sr) -> np.ndarray:
    """Linear chirp from 220 Hz to 880 Hz over 1 second."""
    t = np.linspace(0, DURATION, int(sr * DURATION), endpoint=False)
    f0, f1 = 220.0, 880.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / DURATION * t ** 2)
    return np.sin(phase).astype(np.float32)


@pytest.fixture(scope="session")
def white_noise(sr) -> np.ndarray:
    """White Gaussian noise, unit variance, reproducible seed."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(int(sr * DURATION)).astype(np.float32)


@pytest.fixture(scope="session")
def silent_signal(sr) -> np.ndarray:
    """All-zeros signal."""
    return np.zeros(int(sr * DURATION), dtype=np.float32)


@pytest.fixture(scope="session")
def stft_matrix(sine_440) -> np.ndarray:
    """Pre-computed STFT of the 440 Hz sine for feature tests."""
    from audiolib.core import stft
    return stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH)


@pytest.fixture(scope="session")
def n_fft() -> int:
    return N_FFT


@pytest.fixture(scope="session")
def hop_length() -> int:
    return HOP_LENGTH


# ---------------------------------------------------------------------------
# Helper utilities (not fixtures)
# ---------------------------------------------------------------------------

def assert_shape(arr: np.ndarray, expected_shape: tuple) -> None:
    """Assert array shape matches expected, with clear error message."""
    assert arr.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {arr.shape}"
    )


def assert_allclose_relaxed(a: np.ndarray, b: np.ndarray, rtol: float = 1e-3, atol: float = 1e-5) -> None:
    """Relaxed allclose suited to float32 DSP outputs."""
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def n_frames(n_samples: int, hop_length: int = HOP_LENGTH, n_fft: int = N_FFT, center: bool = True) -> int:
    """Compute expected number of STFT frames."""
    if center:
        n_samples = n_samples + n_fft
    return 1 + (n_samples - n_fft) // hop_length
