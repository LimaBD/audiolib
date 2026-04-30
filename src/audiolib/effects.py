"""
audiolib.effects — Audio effects and transformations.

API-compatible with librosa.effects.
"""
from __future__ import annotations

import numpy as np

from audiolib.core import istft, magphase, stft
from audiolib.exceptions import ParameterError

__all__ = [
    "time_stretch",
    "pitch_shift",
    "split",
    "trim",
    "percussive",
    "harmonic",
    "hpss",
    "remix",
    "preemphasis",
    "deemphasis",
]


def time_stretch(y: np.ndarray, *, rate: float, **kwargs) -> np.ndarray:
    """Time-stretch an audio series by a fixed rate.

    API-compatible with ``librosa.effects.time_stretch``.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Audio time series.
    rate : float > 0
        Stretch factor. ``rate > 1`` speeds up, ``rate < 1`` slows down.

    Returns
    -------
    y_stretch : np.ndarray [shape=(ceil(n/rate),)]
    """
    if rate <= 0:
        raise ParameterError("rate must be positive")

    n_fft = kwargs.get("n_fft", 2048)
    hop_length = kwargs.get("hop_length", n_fft // 4)
    win_length = kwargs.get("win_length")

    D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    n_bins, n_frames = D.shape

    # Phase vocoder: time-scale by interpolating columns
    n_frames_new = int(np.round(n_frames / rate))
    col_idx = np.linspace(0, n_frames - 1, n_frames_new)

    D_stretch = np.zeros((n_bins, n_frames_new), dtype=D.dtype)
    for i, ci in enumerate(col_idx):
        lo = int(ci)
        hi = min(lo + 1, n_frames - 1)
        alpha = ci - lo
        mag_lo, phase_lo = magphase(D[:, lo])
        mag_hi, phase_hi = magphase(D[:, hi])
        mag = (1 - alpha) * mag_lo + alpha * mag_hi
        # Interpolate phase angle
        angle_lo = np.angle(D[:, lo])
        angle_hi = np.angle(D[:, hi])
        angle = (1 - alpha) * angle_lo + alpha * angle_hi
        D_stretch[:, i] = mag * np.exp(1j * angle)

    return istft(D_stretch, hop_length=hop_length, win_length=win_length)


def pitch_shift(
    y: np.ndarray,
    *,
    sr: float,
    n_steps: float,
    bins_per_octave: int = 12,
    res_type: str = "soxr_hq",
    **kwargs,
) -> np.ndarray:
    """Shift the pitch of a waveform by n_steps semitones.

    API-compatible with ``librosa.effects.pitch_shift``.
    """
    from audiolib.core import resample

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    y_shifted = time_stretch(y, rate=rate, **kwargs)
    # Resample back to original length
    return resample(y_shifted, orig_sr=float(sr) / rate, target_sr=float(sr), res_type=res_type)


def trim(
    y: np.ndarray,
    *,
    top_db: float = 60,
    ref=np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim leading and trailing silence from an audio signal.

    API-compatible with ``librosa.effects.trim``.

    Returns
    -------
    y_trimmed : np.ndarray
    index : np.ndarray [shape=(2,)]
        Indices of the non-silent region [start, end).
    """
    from audiolib.core import power_to_db
    from audiolib.feature import rms

    rms_energy = rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = power_to_db(rms_energy ** 2, ref=ref)
    threshold = db.max() - top_db

    is_voiced = db >= threshold
    voiced_idx = np.where(is_voiced)[0]

    if len(voiced_idx) == 0:
        return y[:0], np.array([0, 0])

    start_frame = voiced_idx[0]
    end_frame = voiced_idx[-1] + 1

    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length, len(y))

    return y[start_sample:end_sample], np.array([start_sample, end_sample])


def split(
    y: np.ndarray,
    *,
    top_db: float = 60,
    ref=np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Split an audio signal into non-silent intervals.

    API-compatible with ``librosa.effects.split``.

    Returns
    -------
    intervals : np.ndarray [shape=(n_intervals, 2)]
        Sample indices of non-silent segments.
    """
    from audiolib.core import power_to_db
    from audiolib.feature import rms

    rms_energy = rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = power_to_db(rms_energy ** 2, ref=ref)
    threshold = db.max() - top_db

    is_voiced = db >= threshold
    # Find start/end frame transitions
    padded = np.concatenate([[False], is_voiced, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    intervals = np.column_stack([
        starts * hop_length,
        np.minimum(ends * hop_length, len(y)),
    ])
    return intervals


def harmonic(y: np.ndarray, **kwargs) -> np.ndarray:
    """Extract the harmonic component of a time series.

    API-compatible with ``librosa.effects.harmonic``.
    """
    H, _ = hpss(y, **kwargs)
    return H


def percussive(y: np.ndarray, **kwargs) -> np.ndarray:
    """Extract the percussive component of a time series.

    API-compatible with ``librosa.effects.percussive``.
    """
    _, P = hpss(y, **kwargs)
    return P


def hpss(
    y: np.ndarray,
    *,
    kernel_size: int | tuple[int, int] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: float | tuple[float, float] = 1.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-percussive source separation.

    API-compatible with ``librosa.effects.hpss``.
    """
    try:
        from scipy.ndimage import median_filter
    except ImportError as exc:
        raise ImportError("scipy is required for hpss") from exc

    D = stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)

    if isinstance(kernel_size, int):
        kh = kp = kernel_size
    else:
        kh, kp = kernel_size

    if isinstance(margin, float):
        mh = mp = margin
    else:
        mh, mp = margin

    # Harmonic: filter along time axis (axis=1); Percussive: along freq axis (axis=0)
    H_filter = median_filter(mag, size=(1, kh), mode="reflect")
    P_filter = median_filter(mag, size=(kp, 1), mode="reflect")

    H_filter = H_filter ** power
    P_filter = P_filter ** power

    if mask:
        denom = H_filter + P_filter
        denom[denom < 1e-10] = 1.0
        M_H = H_filter / denom
        M_P = P_filter / denom
        H = M_H * D
        P = M_P * D
    else:
        total = H_filter + P_filter
        total[total < 1e-10] = 1.0
        M_H = H_filter / total
        M_P = P_filter / total
        H = istft(M_H * D, hop_length=hop_length)
        P = istft(M_P * D, hop_length=hop_length)
        return H, P

    return (
        istft(H, hop_length=hop_length),
        istft(P, hop_length=hop_length),
    )


def remix(y: np.ndarray, intervals: np.ndarray, *, align_zeros: bool = True) -> np.ndarray:
    """Remix an audio signal by re-ordering time intervals.

    API-compatible with ``librosa.effects.remix``.
    """
    segments = [y[int(start): int(end)] for start, end in intervals]
    return np.concatenate(segments) if segments else y[:0]


def preemphasis(y: np.ndarray, *, coef: float = 0.97, zi=None, return_zf: bool = False):
    """Apply pre-emphasis to an audio signal.

    API-compatible with ``librosa.effects.preemphasis``.
    """
    from scipy.signal import lfilter, lfilter_zi

    b = np.array([1.0, -coef], dtype=np.float32)
    a = np.array([1.0], dtype=np.float32)

    if zi is None:
        zi = lfilter_zi(b, a) * y[..., :1]

    y_out, zf = lfilter(b, a, y, zi=zi)

    if return_zf:
        return y_out, zf
    return y_out


def deemphasis(y: np.ndarray, *, coef: float = 0.97, zi=None, return_zf: bool = False):
    """Apply de-emphasis to an audio signal.

    API-compatible with ``librosa.effects.deemphasis``.
    """
    from scipy.signal import lfilter, lfilter_zi

    b = np.array([1.0], dtype=np.float32)
    a = np.array([1.0, -coef], dtype=np.float32)

    if zi is None:
        zi = lfilter_zi(b, a) * y[..., :1]

    y_out, zf = lfilter(b, a, y, zi=zi)

    if return_zf:
        return y_out, zf
    return y_out
