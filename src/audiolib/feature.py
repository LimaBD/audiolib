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
from audiolib._core import (
    delta as _delta,
)
from audiolib._core import (
    spectral_contrast as _spectral_contrast,
)
from audiolib._core import (
    poly_features as _poly_features,
)
from audiolib._core import (
    tempogram as _tempogram,
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
    "delta",
    "stack_memory",
    "spectral_contrast",
    "tonnetz",
    "poly_features",
    "tempogram",
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


# ─── Delta ────────────────────────────────────────────────────────────────────


def delta(
    data: np.ndarray,
    *,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: str = "interp",
    **kwargs,
) -> np.ndarray:
    """Compute delta features (local estimate of the derivative).

    API-compatible with ``librosa.feature.delta``.

    Parameters
    ----------
    data : np.ndarray
        Input feature matrix.
    width : int
        Number of frames over which to compute the delta (must be odd and >= 3).
    order : int
        Order of the difference operator (1 = delta, 2 = delta-delta).
    axis : int
        Axis along which to compute deltas (default -1 = time axis).

    Returns
    -------
    delta_data : np.ndarray [same shape as data]
    """
    if order < 1:
        from audiolib.exceptions import ParameterError
        raise ParameterError("order must be at least 1")
    if width < 3 or width % 2 == 0:
        from audiolib.exceptions import ParameterError
        raise ParameterError("width must be odd and >= 3")

    # Work along requested axis by moving it to position -1
    data_moved = np.moveaxis(data, axis, -1)
    orig_shape = data_moved.shape
    n_frames = orig_shape[-1]
    n_features = int(np.prod(orig_shape[:-1]))

    flat = data_moved.astype(np.float32).reshape(n_features, n_frames).flatten().tolist()
    result = _delta(flat, n_features, n_frames, width, order)
    out = np.array(result, dtype=np.float32).reshape(orig_shape)
    return np.moveaxis(out, -1, axis)


# ─── Stack memory ──────────────────────────────────────────────────────────────


def stack_memory(
    data: np.ndarray,
    *,
    n_steps: int = 2,
    delay: int = 1,
    **kwargs,
) -> np.ndarray:
    """Short-term history embedding via memory stacking.

    Vertically stacks ``n_steps`` copies of the data, each shifted by
    ``delay`` frames, creating a (n_features * n_steps, n_frames) matrix.

    API-compatible with ``librosa.feature.stack_memory``.

    Parameters
    ----------
    data : np.ndarray [shape=(d, n_frames)]
    n_steps : int
        Number of steps to stack (including the current frame).
    delay : int
        Number of frames between successive steps.

    Returns
    -------
    data_stacked : np.ndarray [shape=(d * n_steps, n_frames)]
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]

    d, n_frames = data.shape
    out = np.zeros((d * n_steps, n_frames), dtype=data.dtype)

    for step in range(n_steps):
        src_start = max(0, step * delay)
        src_end = n_frames - max(0, (n_steps - step - 1) * delay)
        dst_start_col = max(0, (n_steps - step - 1) * delay)
        dst_end_col = dst_start_col + (src_end - src_start)

        out[step * d : (step + 1) * d, dst_start_col:dst_end_col] = data[:, src_start:src_end]

    return out


# ─── Spectral contrast ────────────────────────────────────────────────────────


def spectral_contrast(
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
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> np.ndarray:
    """Compute spectral contrast.

    API-compatible with ``librosa.feature.spectral_contrast``.

    Returns
    -------
    contrast : np.ndarray [shape=(n_bands + 1, n_frames)]
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _spectral_contrast(
        S_flat,
        float(sr),
        n_fft,
        n_frames,
        n_bins,
        n_bands,
        float(fmin),
        float(quantile),
        linear,
    )
    return np.array(result, dtype=np.float32).reshape(n_bands + 1, n_frames)


# ─── Tonnetz ──────────────────────────────────────────────────────────────────


def tonnetz(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    chroma: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute tonal centroid features (tonnetz).

    Maps a chromagram to a 6-dimensional tonal centroid representation.

    API-compatible with ``librosa.feature.tonnetz``.

    Returns
    -------
    tonnetz : np.ndarray [shape=(6, n_frames)]
    """
    if chroma is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or chroma must be provided")
        chroma = chroma_stft(y=y, sr=sr)

    # Transformation matrix: maps 12 chroma bins to 6 tonal centroid dims
    # Rows: fifths-x, fifths-y, minor-x, minor-y, major-x, major-y
    pitch_class = np.arange(12, dtype=np.float32)
    r = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Angles for each transformation (in radians)
    angles = np.array(
        [
            7.0 / 6.0,   # fifths
            7.0 / 6.0,
            3.0 / 2.0,   # minor
            3.0 / 2.0,
            2.0 / 3.0,   # major
            2.0 / 3.0,
        ],
        dtype=np.float32,
    )

    phase_offsets = np.array([0.0, np.pi / 2, 0.0, np.pi / 2, 0.0, np.pi / 2], dtype=np.float32)

    T = np.zeros((6, 12), dtype=np.float32)
    for i in range(6):
        T[i] = np.cos(2.0 * np.pi * pitch_class * angles[i] / 12.0 + phase_offsets[i])

    # Normalize chroma columns
    chroma_norm = chroma.astype(np.float32)
    col_sums = chroma_norm.sum(axis=0, keepdims=True)
    col_sums[col_sums < 1e-10] = 1.0
    chroma_norm = chroma_norm / col_sums

    return T @ chroma_norm  # (6, n_frames)


# ─── Poly features ───────────────────────────────────────────────────────────


def poly_features(
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
    order: int = 1,
    freq=None,
) -> np.ndarray:
    """Fit a polynomial to the spectral columns.

    API-compatible with ``librosa.feature.poly_features``.

    Parameters
    ----------
    order : int
        Polynomial degree (default 1 = linear fit).

    Returns
    -------
    poly_features : np.ndarray [shape=(order + 1, n_frames)]
        Coefficient matrix, highest-degree first.
    """
    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center)
        S = np.abs(D)

    n_bins, n_frames = S.shape
    S_flat = S.astype(np.float32).T.flatten().tolist()
    result = _poly_features(S_flat, float(sr), n_fft, n_frames, n_bins, order)
    return np.array(result, dtype=np.float32).reshape(order + 1, n_frames)


# ─── Tempogram ────────────────────────────────────────────────────────────────


def tempogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: str = "hann",
    norm: Optional[float] = np.inf,
) -> np.ndarray:
    """Compute a tempogram (local autocorrelation of the onset strength).

    API-compatible with ``librosa.feature.tempogram``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio time series.
    sr : float
        Sampling rate.
    onset_envelope : np.ndarray or None
        Pre-computed onset strength envelope.
    hop_length : int
        Hop length used for the onset envelope.
    win_length : int
        Number of frames for the window (default 384).
    center : bool
        Pad the onset envelope if True.
    norm : float or None
        Column-wise normalisation. ``np.inf`` → max norm, ``None`` → no norm.

    Returns
    -------
    tempogram : np.ndarray [shape=(win_length, n_frames)]
    """
    if onset_envelope is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or onset_envelope must be provided")
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    oenv = onset_envelope.astype(np.float32).tolist()
    flat, n_tempo_bins, n_tg_frames = _tempogram(
        oenv, float(sr), int(hop_length), int(win_length), center
    )
    tg = np.array(flat, dtype=np.float32).reshape(n_tempo_bins, n_tg_frames)

    if norm is not None:
        if norm == np.inf:
            col_max = np.max(np.abs(tg), axis=0, keepdims=True)
            col_max[col_max < 1e-10] = 1.0
            tg = tg / col_max
        elif norm > 0:
            col_norms = np.linalg.norm(tg, ord=norm, axis=0, keepdims=True)
            col_norms[col_norms < 1e-10] = 1.0
            tg = tg / col_norms

    return tg
