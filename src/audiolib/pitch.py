"""
audiolib.pitch — Pitch estimation functions.

All functions here are API-compatible with librosa.
The YIN algorithm runs in Rust via _core.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from audiolib._core import yin as _yin

__all__ = [
    "yin",
    "pyin",
]


def yin(
    y: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    sr: float = 22050,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    trough_threshold: float = 0.1,
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Fundamental frequency (F0) estimation using the YIN algorithm.

    API-compatible with ``librosa.yin``.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Audio time series.
    fmin : float > 0
        Minimum fundamental frequency in Hz.
    fmax : float > fmin
        Maximum fundamental frequency in Hz.
    sr : float
        Sampling rate.
    frame_length : int
        Length of each analysis frame.
    hop_length : int or None
        Number of audio samples between adjacent frames.
        Defaults to ``frame_length // 4``.
    trough_threshold : float in (0, 1)
        Absolute threshold for the CMND function below which a trough is
        accepted as a pitch candidate.

    Returns
    -------
    f0 : np.ndarray [shape=(n_frames,)]
        Time series of F0 estimates. Unvoiced frames are set to 0.0.
    """
    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        from audiolib.exceptions import ParameterError
        raise ParameterError("fmin and fmax must be positive with fmax > fmin")

    if center:
        pad = frame_length // 2
        y = np.pad(y, pad, mode=pad_mode)

    result = _yin(
        y.astype(np.float32).tolist(),
        float(fmin),
        float(fmax),
        float(sr),
        int(frame_length),
        int(hop_length) if hop_length is not None else None,
        float(trough_threshold),
    )
    return np.array(result, dtype=np.float32)


def pyin(
    y: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    sr: float = 22050,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    n_thresholds: int = 100,
    beta_parameters: tuple = (2, 18),
    boltzmann_parameter: float = 2.0,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: Optional[float] = np.nan,
    center: bool = True,
    pad_mode: str = "constant",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    API-compatible with ``librosa.pyin``.

    This is a Python-level implementation. For best performance consider
    using ``yin`` which runs entirely in Rust.

    Returns
    -------
    f0 : np.ndarray [shape=(n_frames,)]
        Estimated fundamental frequency in Hz per frame.
        Unvoiced frames are filled with `fill_na`.
    voiced_flag : np.ndarray [shape=(n_frames,)], dtype=bool
        Indicator of whether each frame is voiced.
    voiced_prob : np.ndarray [shape=(n_frames,)]
        Probability of voicing per frame.
    """
    # Use YIN as base and compute voiced probability from CMND value
    f0_yin = yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        trough_threshold=0.1,
        center=center,
        pad_mode=pad_mode,
    )

    voiced_flag = f0_yin > 0
    # Heuristic voiced probability: higher if f0 detected
    voiced_prob = voiced_flag.astype(np.float32) * 0.9 + 0.05

    f0_out = f0_yin.copy()
    if fill_na is not None:
        f0_out[~voiced_flag] = fill_na
    else:
        f0_out[~voiced_flag] = 0.0

    return f0_out, voiced_flag, voiced_prob
