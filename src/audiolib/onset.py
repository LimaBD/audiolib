"""
audiolib.onset — Onset detection functions.

All functions here are API-compatible with librosa.onset.
The hot path runs in Rust via _core.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from audiolib._core import onset_detect as _onset_detect

__all__ = [
    "onset_detect",
    "onset_strength",
    "onset_strength_multi",
]


def onset_detect(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    backtrack: bool = False,
    normalize: bool = True,
    energy: Optional[np.ndarray] = None,
    units: str = "time",
    delta: float = 0.07,
    wait: int = 30,
    **kwargs,
) -> np.ndarray:
    """Locate note onset events from an audio series or onset strength.

    API-compatible with ``librosa.onset.onset_detect``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio time series.
    sr : float
        Sampling rate.
    onset_envelope : np.ndarray or None
        Pre-computed onset strength envelope.
    hop_length : int
        Hop length used when computing onset envelope.
    units : str
        Output unit for onset positions: ``'frames'``, ``'samples'``, or ``'time'``.
    delta : float
        Threshold above the local mean for onset detection.
    wait : int
        Minimum number of frames between adjacent onsets.

    Returns
    -------
    onsets : np.ndarray [shape=(n_onsets,)]
        Onset positions in the requested units.
    """
    if onset_envelope is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or onset_envelope must be provided")
        from audiolib.feature import onset_strength as _onset_strength
        onset_envelope = _onset_strength(y=y, sr=sr, hop_length=hop_length, **kwargs)

    oenv = onset_envelope.astype(np.float32).tolist()
    onset_frames = np.asarray(
        _onset_detect(oenv, float(sr), int(hop_length), float(delta), int(wait)),
        dtype=np.int32,
    )

    if units == "frames":
        return onset_frames
    elif units == "samples":
        return onset_frames * hop_length
    elif units == "time":
        return onset_frames * hop_length / sr
    else:
        from audiolib.exceptions import ParameterError
        raise ParameterError(f"Unknown units: {units!r}. Use 'frames', 'samples', or 'time'.")


def onset_strength(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    hop_length: int = 512,
    **kwargs,
) -> np.ndarray:
    """Compute a spectral flux onset strength envelope.

    Alias for ``audiolib.feature.onset_strength``.
    API-compatible with ``librosa.onset.onset_strength``.
    """
    from audiolib.feature import onset_strength as _feat_onset
    return _feat_onset(y=y, sr=sr, S=S, hop_length=hop_length, **kwargs)


def onset_strength_multi(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_mels: int = 138,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    channels=None,
    **kwargs,
) -> np.ndarray:
    """Multi-channel spectral flux onset strength envelopes.

    API-compatible with ``librosa.onset.onset_strength_multi``.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(n_channels, n_frames)]
    """
    from audiolib.core import stft
    from audiolib.feature import melspectrogram, onset_strength as _feat_onset

    if S is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or S must be provided")
        S = np.abs(stft(y, hop_length=hop_length))

    # Split into mel sub-bands then compute flux per band
    sr_use = float(sr) if sr else 22050.0
    n_fft = (S.shape[0] - 1) * 2

    mel_S = melspectrogram(S=S ** 2, sr=sr_use, n_fft=n_fft, n_mels=n_mels)
    # Split into frequency channels
    if channels is None:
        channels = [range(n_mels)]

    result = []
    for ch in channels:
        ch_S = mel_S[list(ch), :]
        oenv = _feat_onset(S=ch_S, sr=sr_use, hop_length=hop_length)
        result.append(oenv)

    return np.array(result, dtype=np.float32)
