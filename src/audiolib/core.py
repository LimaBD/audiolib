"""
audiolib.core — Core audio I/O and DSP functions.

All functions here are API-compatible with librosa.core.
The hot path (STFT, filterbanks, feature extraction) runs in Rust via _core.
Audio file loading delegates to soundfile for broad codec support.
"""
from __future__ import annotations

import math
import os
from typing import BinaryIO

import numpy as np
import soundfile as sf

from audiolib._core import (
    amplitude_to_db as _amp_to_db,
)
from audiolib._core import (
    autocorrelate as _autocorrelate,
)
from audiolib._core import (
    db_to_amplitude as _db_to_amp,
)
from audiolib._core import (
    db_to_power as _db_to_pow,
)
from audiolib._core import (
    istft as _istft_rust,
)
from audiolib._core import (
    mu_compress as _mu_compress,
)
from audiolib._core import (
    mu_expand as _mu_expand,
)
from audiolib._core import (
    power_to_db as _pow_to_db,
)
from audiolib._core import (
    resample as _resample,
)
from audiolib._core import (
    stft as _stft_rust,
)
from audiolib._core import (
    to_mono as _to_mono,
)
from audiolib._core import (
    zero_crossings as _zero_crossings,
)
from audiolib.exceptions import ParameterError

__all__ = [
    "load",
    "get_duration",
    "get_samplerate",
    "to_mono",
    "resample",
    "stft",
    "istft",
    "magphase",
    "amplitude_to_db",
    "db_to_amplitude",
    "power_to_db",
    "db_to_power",
    "zero_crossings",
    "autocorrelate",
    "mu_compress",
    "mu_expand",
    "clicks",
    "tone",
    "chirp",
]

# ─── Audio loading ────────────────────────────────────────────────────────────


def load(
    path: str | int | os.PathLike | BinaryIO,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    dtype=np.float32,
    res_type: str = "soxr_hq",
) -> tuple[np.ndarray, int | float]:
    """Load an audio file as a floating point time series.

    API-compatible with ``librosa.load``.

    Parameters
    ----------
    path : str, pathlib.Path, or file-like object
        Path to the audio file.
    sr : int or None
        Target sampling rate. ``None`` preserves the native sample rate.
    mono : bool
        Convert to mono by averaging channels.
    offset : float
        Start reading after this time (in seconds).
    duration : float or None
        Only load up to this much audio (seconds).
    dtype : dtype
        NumPy dtype for the output array (default float32).
    res_type : str
        Resampling method (see ``resample``). Default ``'soxr_hq'``.

    Returns
    -------
    y : np.ndarray
        Audio time series.
    sr : int
        Sampling rate.
    """
    with sf.SoundFile(path) as f:
        sr_native = f.samplerate

        if offset > 0.0:
            f.seek(int(offset * sr_native))

        frames = int(duration * sr_native) if duration is not None else -1

        y = f.read(frames=frames, dtype=dtype, always_2d=True).T  # (channels, samples)

    if mono and y.shape[0] > 1:
        y = to_mono(y)
    elif y.shape[0] == 1:
        y = y[0]

    if sr is not None and int(sr) != sr_native:
        y = resample(y, orig_sr=float(sr_native), target_sr=float(sr), res_type=res_type)
    else:
        sr = sr_native

    return y, int(sr)


def get_duration(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    path: str | os.PathLike | None = None,
) -> float:
    """Compute the duration (in seconds) of an audio time series or file.

    API-compatible with ``librosa.get_duration``.
    """
    if path is not None:
        return float(sf.info(path).duration)

    if y is not None:
        return float(y.shape[-1]) / sr

    if S is not None:
        n_frames = S.shape[-1]
        n_samples = n_fft + hop_length * (n_frames - 1)
        if center:
            n_samples -= 2 * (n_fft // 2)
        return float(n_samples) / sr

    raise ParameterError("At least one of (y, sr), S, or path must be provided")


def get_samplerate(path: str | int | BinaryIO) -> float:
    """Get the sampling rate for a given file.

    API-compatible with ``librosa.get_samplerate``.
    """
    return float(sf.info(path).samplerate)


def to_mono(y: np.ndarray) -> np.ndarray:
    """Convert an audio signal to mono.

    API-compatible with ``librosa.to_mono``.
    """
    if y.ndim == 1:
        return y

    # Use Rust for contiguous float32 arrays
    if y.dtype == np.float32 and y.data.c_contiguous and y.ndim == 2:
        n_channels, n_samples = y.shape
        flat = y.flatten().tolist()
        result = _to_mono(flat, n_channels)
        return np.array(result, dtype=np.float32)

    # Fallback for other dtypes/shapes
    return np.mean(y, axis=tuple(range(y.ndim - 1))).astype(y.dtype)


def resample(
    y: np.ndarray,
    *,
    orig_sr: float,
    target_sr: float,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Resample a time series from orig_sr to target_sr.

    API-compatible with ``librosa.resample``.
    Uses soxr when available for high-quality resampling, with Rust linear
    interpolation as a lightweight fallback.
    """
    if orig_sr == target_sr:
        return y

    # Try soxr for high-quality resampling (available on most platforms)
    try:
        import soxr
        y_hat = np.apply_along_axis(
            soxr.resample,
            axis=axis,
            arr=y,
            in_rate=orig_sr,
            out_rate=target_sr,
            quality=res_type if res_type.startswith("soxr") else "soxr_hq",
        )
    except ImportError:
        # Fallback to Rust linear interpolation
        y_hat = _resample_via_rust(y, orig_sr, target_sr, axis)

    if fix:
        n_samples = int(math.ceil(y.shape[axis] * float(target_sr) / orig_sr))
        y_hat = _fix_length(y_hat, n_samples, axis=axis)

    if scale:
        y_hat = y_hat / math.sqrt(float(target_sr) / orig_sr)

    return np.asarray(y_hat, dtype=y.dtype)


def _resample_via_rust(y: np.ndarray, orig_sr: float, target_sr: float, axis: int) -> np.ndarray:
    """Internal: use Rust linear resampler along a given axis."""
    if y.ndim == 1:
        flat = y.astype(np.float32).tolist()
        result = _resample(flat, float(orig_sr), float(target_sr))
        return np.array(result, dtype=np.float32)
    return np.apply_along_axis(
        lambda row: np.array(_resample(row.astype(np.float32).tolist(), float(orig_sr), float(target_sr)), dtype=np.float32),
        axis=axis,
        arr=y,
    )


def _fix_length(y: np.ndarray, size: int, axis: int = -1) -> np.ndarray:
    """Pad or trim y along axis to exactly 'size' samples."""
    current = y.shape[axis]
    if current == size:
        return y
    if current > size:
        idx = [slice(None)] * y.ndim
        idx[axis] = slice(size)
        return y[tuple(idx)]
    # Pad
    pad_width = [(0, 0)] * y.ndim
    pad_width[axis] = (0, size - current)
    return np.pad(y, pad_width)


# ─── STFT / ISTFT ─────────────────────────────────────────────────────────────


def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    dtype=np.complex64,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Short-time Fourier transform (STFT).

    API-compatible with ``librosa.stft``.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Audio time series. Must be mono.
    n_fft : int
        FFT window size.
    hop_length : int or None
        Number of samples between frames (default n_fft // 4).
    win_length : int or None
        Window length (default = n_fft).
    center : bool
        Pad signal at both ends if True.

    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=complex64]
        Complex STFT matrix.
    """
    if y.ndim != 1:
        raise ParameterError("y must be 1-D (use to_mono first)")

    hop = hop_length if hop_length is not None else n_fft // 4
    padded, n_frames, n_bins = _stft_rust(
        y.astype(np.float32).tolist(),
        n_fft,
        hop,
        win_length,
        center,
    )
    if n_frames == 0:
        return np.zeros((n_bins, 0), dtype=dtype)

    # Reshape flat output from (n_frames, n_bins, 2) to complex (n_bins, n_frames)
    arr = np.array(padded, dtype=np.float32).reshape(n_frames, n_bins, 2)
    D = arr[:, :, 0] + 1j * arr[:, :, 1]  # (n_frames, n_bins)
    return D.T.astype(dtype)  # (n_bins, n_frames)


def istft(
    stft_matrix: np.ndarray,
    *,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    dtype=np.float32,
    length: int | None = None,
) -> np.ndarray:
    """Inverse short-time Fourier transform.

    API-compatible with ``librosa.istft``.
    """
    n_bins, n_frames = stft_matrix.shape
    n_fft = (n_bins - 1) * 2
    hop = hop_length if hop_length is not None else n_fft // 4
    win_len = win_length if win_length is not None else n_fft

    # Flatten to (n_frames, n_bins) for Rust
    mat = stft_matrix.T.astype(np.complex64)  # (n_frames, n_bins)
    re = mat.real.flatten().tolist()
    im = mat.imag.flatten().tolist()

    result = _istft_rust(re, im, n_frames, n_bins, hop, win_len, center)
    y = np.array(result, dtype=dtype)

    if length is not None:
        y = _fix_length(y, length)

    return y


def magphase(D: np.ndarray, *, power: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """Separate a complex STFT into magnitude and phase.

    API-compatible with ``librosa.magphase``.

    Returns
    -------
    magnitude, phase : np.ndarray
    """
    mag = np.abs(D) ** power
    phase = np.exp(1j * np.angle(D))
    return mag, phase


# ─── Magnitude scaling ────────────────────────────────────────────────────────


def amplitude_to_db(
    S: np.ndarray,
    *,
    ref: float | callable = 1.0,
    amin: float = 1e-5,
    top_db: float | None = 80.0,
) -> np.ndarray:
    """Convert amplitude spectrogram to dB.

    API-compatible with ``librosa.amplitude_to_db``.
    """
    ref_val = float(ref(np.abs(S))) if callable(ref) else float(ref)
    flat = S.astype(np.float32).flatten().tolist()
    result = _amp_to_db(flat, ref_val, float(amin), top_db)
    return np.array(result, dtype=np.float32).reshape(S.shape)


def power_to_db(
    S: np.ndarray,
    *,
    ref: float | callable = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.ndarray:
    """Convert power spectrogram to dB.

    API-compatible with ``librosa.power_to_db``.
    """
    ref_val = float(ref(S)) if callable(ref) else float(ref)
    flat = S.astype(np.float32).flatten().tolist()
    result = _pow_to_db(flat, ref_val, float(amin), top_db)
    return np.array(result, dtype=np.float32).reshape(S.shape)


def db_to_amplitude(S_db: np.ndarray, *, ref: float = 1.0) -> np.ndarray:
    """Convert dB-scaled spectrogram to amplitude.

    API-compatible with ``librosa.db_to_amplitude``.
    """
    flat = S_db.astype(np.float32).flatten().tolist()
    result = _db_to_amp(flat, float(ref))
    return np.array(result, dtype=np.float32).reshape(S_db.shape)


def db_to_power(S_db: np.ndarray, *, ref: float = 1.0) -> np.ndarray:
    """Convert dB-scale to power.

    API-compatible with ``librosa.db_to_power``.
    """
    flat = S_db.astype(np.float32).flatten().tolist()
    result = _db_to_pow(flat, float(ref))
    return np.array(result, dtype=np.float32).reshape(S_db.shape)


# ─── Time-domain processing ───────────────────────────────────────────────────


def zero_crossings(
    y: np.ndarray,
    *,
    threshold: float = 1e-10,
    ref_magnitude=None,
    pad: bool = True,
    zero_pos: bool = True,
    axis: int = -1,
) -> np.ndarray:
    """Find the zero-crossings of a signal.

    API-compatible with ``librosa.zero_crossings``.
    """
    if ref_magnitude is not None:
        if callable(ref_magnitude):
            threshold = threshold * ref_magnitude(np.abs(y))
        else:
            threshold = threshold * ref_magnitude

    flat = y.swapaxes(axis, -1).astype(np.float32)
    orig_shape = flat.shape
    flat = flat.reshape(-1, orig_shape[-1])

    out_rows = []
    for row in flat:
        out_rows.append(_zero_crossings(row.tolist(), float(threshold), pad))

    result = np.array(out_rows, dtype=bool).reshape(orig_shape)
    return np.swapaxes(result, -1, axis)


def autocorrelate(
    y: np.ndarray,
    *,
    max_size: int | None = None,
    axis: int = -1,
) -> np.ndarray:
    """Bounded-lag auto-correlation.

    API-compatible with ``librosa.autocorrelate``.
    """
    y_swap = np.swapaxes(y, axis, -1)
    orig_shape = y_swap.shape
    flat = y_swap.reshape(-1, orig_shape[-1])

    rows = []
    for row in flat:
        rows.append(_autocorrelate(row.astype(np.float32).tolist(), max_size))

    out_last_dim = max_size if max_size is not None else orig_shape[-1]
    out_shape = orig_shape[:-1] + (min(out_last_dim, orig_shape[-1]),)
    result = np.array(rows, dtype=np.float32).reshape(out_shape)
    return np.swapaxes(result, -1, axis)


def mu_compress(
    x: np.ndarray,
    *,
    mu: float = 255.0,
    quantize: bool = False,
) -> np.ndarray:
    """mu-law compression.

    API-compatible with ``librosa.mu_compress``.
    """
    flat = x.astype(np.float32).flatten().tolist()
    result = _mu_compress(flat, float(mu), quantize)
    return np.array(result, dtype=np.float32 if not quantize else np.int16).reshape(x.shape)


def mu_expand(
    x,
    *,
    mu: float = 255.0,
    quantize: bool = False,
) -> np.ndarray:
    """mu-law expansion.

    API-compatible with ``librosa.mu_expand``.
    """
    arr = np.asarray(x, dtype=np.float32)
    flat = arr.flatten().tolist()
    result = _mu_expand(flat, float(mu), quantize)
    return np.array(result, dtype=np.float32).reshape(arr.shape)


# ─── Signal generation ────────────────────────────────────────────────────────


def clicks(
    *,
    times=None,
    frames=None,
    sr: float = 22050,
    hop_length: int = 512,
    click_freq: float = 1000.0,
    click_duration: float = 0.1,
    click=None,
    length: int | None = None,
) -> np.ndarray:
    """Construct a click track.

    API-compatible with ``librosa.clicks``.
    """
    from audiolib.convert import frames_to_samples, time_to_samples

    if times is not None:
        positions = time_to_samples(np.asarray(times, dtype=np.float64).tolist(), sr=float(sr))
    elif frames is not None:
        positions = frames_to_samples(np.asarray(frames, dtype=np.int64).tolist(), hop_length)
    else:
        raise ParameterError('either "times" or "frames" must be provided')

    positions = np.array(positions, dtype=np.int64)

    if click is not None:
        click_signal_template = np.asarray(click, dtype=np.float32)
    else:
        if click_duration <= 0:
            raise ParameterError("click_duration must be strictly positive")
        if click_freq <= 0:
            raise ParameterError("click_freq must be strictly positive")
        n_click = int(sr * click_duration)
        t = np.arange(n_click)
        envelope = np.logspace(0, -10, num=n_click, base=2.0)
        click_signal_template = (envelope * np.sin(2 * np.pi * click_freq / sr * t)).astype(np.float32)

    if length is None:
        length = int(positions.max()) + len(click_signal_template) if len(positions) > 0 else 0

    out = np.zeros(length, dtype=np.float32)
    for start in positions:
        end = int(start) + len(click_signal_template)
        if end >= length:
            out[int(start):] += click_signal_template[: length - int(start)]
        else:
            out[int(start):end] += click_signal_template

    return out


def tone(
    frequency: float,
    *,
    sr: float = 22050,
    length: int | None = None,
    duration: float | None = None,
    phi: float | None = None,
) -> np.ndarray:
    """Construct a pure tone (cosine) signal.

    API-compatible with ``librosa.tone``.
    """
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
        length = int(duration * sr)

    if phi is None:
        phi = -np.pi * 0.5

    return np.cos(2 * np.pi * frequency * np.arange(length) / sr + phi).astype(np.float32)


def chirp(
    *,
    fmin: float,
    fmax: float,
    sr: float = 22050,
    length: int | None = None,
    duration: float | None = None,
    linear: bool = False,
    phi: float | None = None,
) -> np.ndarray:
    """Construct a chirp / sine-sweep signal.

    API-compatible with ``librosa.chirp``.
    """
    try:
        import scipy.signal as sig
    except ImportError as exc:
        raise ImportError("scipy is required for chirp generation") from exc

    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')

    period = 1.0 / sr
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
    else:
        duration = period * length

    if phi is None:
        phi = -np.pi * 0.5

    method = "linear" if linear else "logarithmic"
    return sig.chirp(
        np.arange(int(duration * sr)) / sr,
        fmin,
        duration,
        fmax,
        method=method,
        phi=phi / np.pi * 180,
    ).astype(np.float32)
