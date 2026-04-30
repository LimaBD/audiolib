"""
audiolib.feature — Feature extraction functions.

All functions here are API-compatible with librosa.feature.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from audiolib._core import (
    chroma_stft as _chroma_stft,
)
from audiolib._core import (
    get_rms as _get_rms,
)
from audiolib._core import (
    mel_filterbank as _mel_filterbank,
)
from audiolib._core import (
    melspectrogram as _melspectrogram,
)
from audiolib._core import (
    mfcc as _mfcc,
)
from audiolib._core import (
    onset_strength as _onset_strength,
)
from audiolib._core import (
    spectral_bandwidth as _spectral_bandwidth,
)
from audiolib._core import (
    spectral_centroid as _spectral_centroid,
)
from audiolib._core import (
    spectral_flatness as _spectral_flatness,
)
from audiolib._core import (
    spectral_rolloff as _spectral_rolloff,
)
from audiolib.core import power_to_db, stft

__all__ = [
    "melspectrogram",
    "mfcc",
    "chroma_stft",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "rms",
    "zero_crossing_rate",
    "onset_strength",
]

# ─── Mel spectrogram ──────────────────────────────────────────────────────────


def melspectrogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
) -> np.ndarray:
    """Compute a mel spectrogram.

    API-compatible with ``librosa.feature.melspectrogram``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio time series.
    sr : float
        Sampling rate.
    S : np.ndarray or None
        Pre-computed power spectrogram (skips STFT if provided).
    n_fft, hop_length, win_length, center, power : standard STFT params.
    n_mels : int
        Number of mel bands.
    fmin, fmax : float
        Frequency range for mel filterbank.
    htk : bool
        Use HTK mel formula instead of Slaney.

    Returns
    -------
    M : np.ndarray [shape=(n_mels, n_frames)]
        Mel spectrogram.
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D) ** power  # power spectrogram

    n_bins, n_frames = S.shape

    # Build mel filterbank in Rust
    fb = _mel_filterbank(float(sr), n_fft, n_mels, float(fmin), fmax, htk)
    fb_arr = np.array(fb, dtype=np.float32).reshape(n_mels, n_bins)

    # Compute mel spectrogram in Rust
    S_flat = S.astype(np.float32).T.flatten().tolist()  # (n_frames, n_bins)
    result = _melspectrogram(S_flat, fb_arr.flatten().tolist(), n_frames, n_bins, n_mels)

    return np.array(result, dtype=np.float32).reshape(n_mels, n_frames)


# ─── MFCCs ────────────────────────────────────────────────────────────────────


def mfcc(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: int = 0,
    n_mels: int = 128,
    **kwargs,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients (MFCCs).

    API-compatible with ``librosa.feature.mfcc``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio time series.
    sr : float
        Sampling rate.
    S : np.ndarray or None
        Pre-computed log-power mel spectrogram.
    n_mfcc : int
        Number of MFCCs to return.
    n_mels : int
        Number of mel bands.

    Returns
    -------
    M : np.ndarray [shape=(n_mfcc, n_frames)]
        MFCC matrix.
    """
    if S is None:
        mel = melspectrogram(y=y, sr=sr, n_mels=n_mels, **kwargs)
        S = power_to_db(mel)

    n_mels_real, n_frames = S.shape
    flat = S.astype(np.float32).flatten().tolist()  # row-major: (n_mels, n_frames)
    result = _mfcc(flat, n_mels_real, n_frames, n_mfcc)
    out = np.array(result, dtype=np.float32).reshape(n_mfcc, n_frames)

    if norm == "ortho":
        out[0] *= np.sqrt(1.0 / (4 * n_mels_real))
        out[1:] *= np.sqrt(1.0 / (2 * n_mels_real))

    if lifter > 0:
        li = 1.0 + (lifter / 2.0) * np.sin(np.pi * np.arange(1, n_mfcc + 1) / lifter)
        out *= li[:, np.newaxis]

    return out


# ─── Chroma ───────────────────────────────────────────────────────────────────


def chroma_stft(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    norm: Optional[float] = np.inf,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    tuning: float = 0.0,
    n_chroma: int = 12,
) -> np.ndarray:
    """Compute a chromagram from a waveform or power spectrogram.

    API-compatible with ``librosa.feature.chroma_stft``.

    Returns
    -------
    chroma : np.ndarray [shape=(n_chroma, n_frames)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()  # (n_frames, n_bins)
    result = _chroma_stft(S_flat, float(sr), n_fft, n_frames, n_bins, tuning)

    chroma = np.array(result, dtype=np.float32).reshape(12, n_frames)

    if norm is not None and norm == np.inf:
        col_max = np.max(np.abs(chroma), axis=0, keepdims=True)
        col_max[col_max < 1e-10] = 1.0
        chroma /= col_max

    return chroma


# ─── Spectral features ────────────────────────────────────────────────────────


def spectral_centroid(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq=None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Compute the spectral centroid.

    API-compatible with ``librosa.feature.spectral_centroid``.

    Returns
    -------
    centroid : np.ndarray [shape=(1, n_frames)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _spectral_centroid(S_flat, float(sr), n_fft, n_frames, n_bins)
    return np.array(result, dtype=np.float32).reshape(1, n_frames)


def spectral_bandwidth(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq=None,
    centroid=None,
    norm: bool = True,
    p: float = 2,
) -> np.ndarray:
    """Compute the spectral bandwidth.

    API-compatible with ``librosa.feature.spectral_bandwidth``.

    Returns
    -------
    bandwidth : np.ndarray [shape=(1, n_frames)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _spectral_bandwidth(S_flat, float(sr), n_fft, n_frames, n_bins, float(p))
    return np.array(result, dtype=np.float32).reshape(1, n_frames)


def spectral_rolloff(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq=None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    roll_percent: float = 0.85,
) -> np.ndarray:
    """Compute the spectral rolloff frequency.

    API-compatible with ``librosa.feature.spectral_rolloff``.

    Returns
    -------
    rolloff : np.ndarray [shape=(1, n_frames)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _spectral_rolloff(S_flat, float(sr), n_fft, n_frames, n_bins, float(roll_percent))
    return np.array(result, dtype=np.float32).reshape(1, n_frames)


def spectral_flatness(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    amin: float = 1e-10,
    power: float = 2.0,
) -> np.ndarray:
    """Compute the spectral flatness.

    API-compatible with ``librosa.feature.spectral_flatness``.

    Returns
    -------
    flatness : np.ndarray [shape=(1, n_frames)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D) ** power

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _spectral_flatness(S_flat, n_frames, n_bins)
    return np.array(result, dtype=np.float32).reshape(1, n_frames)


# ─── RMS ─────────────────────────────────────────────────────────────────────


def rms(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Compute root-mean-square (RMS) energy.

    API-compatible with ``librosa.feature.rms``.

    Returns
    -------
    rms : np.ndarray [shape=(1, n_frames)]
    """
    if y is not None:
        result = _get_rms(y.astype(np.float32).tolist(), frame_length, hop_length)
        return np.array(result, dtype=np.float32).reshape(1, -1)

    if S is not None:
        return np.sqrt(np.mean(np.abs(S) ** 2, axis=0, keepdims=True)).astype(np.float32)

    from audiolib.exceptions import ParameterError
    raise ParameterError("one of y or S must be provided")


# ─── Zero crossing rate ───────────────────────────────────────────────────────


def zero_crossing_rate(
    y: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
) -> np.ndarray:
    """Compute the zero-crossing rate.

    API-compatible with ``librosa.feature.zero_crossing_rate``.

    Returns
    -------
    zcr : np.ndarray [shape=(1, n_frames)]
    """
    from audiolib.core import zero_crossings

    if center:
        pad = frame_length // 2
        y = np.pad(y, pad)

    n = y.shape[-1]
    n_frames = 1 + (n - frame_length) // hop_length if n >= frame_length else 0

    zcr = np.zeros((1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start: start + frame_length]
        zc = zero_crossings(frame)
        zcr[0, i] = float(np.sum(zc)) / frame_length

    return zcr


# ─── Onset strength ───────────────────────────────────────────────────────────


def onset_strength(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    aggregate=np.mean,
    **kwargs,
) -> np.ndarray:
    """Compute a spectral flux onset strength envelope.

    API-compatible with ``librosa.onset.onset_strength``.

    Returns
    -------
    onset_env : np.ndarray [shape=(n_frames,)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _onset_strength(S_flat, n_frames, n_bins)
    return np.array(result, dtype=np.float32)
