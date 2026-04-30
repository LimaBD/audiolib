"""
audiolib.beat — Beat tracking and tempo estimation.

All functions here are API-compatible with librosa.beat.
The hot path runs in Rust via _core.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from audiolib._core import (
    beat_tempo as _beat_tempo,
)
from audiolib._core import (
    beat_track_dp as _beat_track_dp,
)

__all__ = [
    "beat_track",
    "tempo",
]


def tempo(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: float = 320.0,
    aggregate=np.mean,
    prior=None,
) -> np.ndarray:
    """Estimate the global tempo (BPM) from an audio signal or onset envelope.

    API-compatible with ``librosa.beat.tempo``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio time series.
    sr : float
        Sampling rate.
    onset_envelope : np.ndarray or None
        Pre-computed onset strength envelope.
    hop_length : int
        Hop length used for onset envelope computation.
    start_bpm : float
        Initial guess for tempo in BPM (prior center).
    max_tempo : float
        Maximum credible tempo (BPM).

    Returns
    -------
    tempo : np.ndarray [shape=()]
        Estimated global tempo in BPM.
    """
    if onset_envelope is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or onset_envelope must be provided")
        from audiolib.feature import onset_strength
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    oenv = onset_envelope.astype(np.float32).tolist()
    bpm = _beat_tempo(
        oenv,
        float(sr),
        int(hop_length),
        float(start_bpm),
        float(max_tempo),
    )
    return np.array([bpm], dtype=np.float32)


def beat_track(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100.0,
    trim: bool = True,
    bpm: Optional[float] = None,
    prior=None,
    units: str = "frames",
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic programming beat tracker.

    API-compatible with ``librosa.beat.beat_track``.

    Parameters
    ----------
    y : np.ndarray or None
        Audio time series.
    sr : float
        Sampling rate.
    onset_envelope : np.ndarray or None
        Pre-computed onset strength envelope.
    hop_length : int
        Hop length in samples.
    start_bpm : float
        Initial tempo estimate in BPM.
    tightness : float
        Tightness of the beat distribution.
    trim : bool
        Trim low-onset beats from start/end.
    bpm : float or None
        Override the tempo with this value (skips tempo estimation).
    units : str
        Output representation for beat positions.
        ``'frames'`` (default), ``'samples'``, or ``'time'``.

    Returns
    -------
    tempo : np.ndarray [scalar]
        Estimated tempo in BPM.
    beats : np.ndarray [shape=(n_beats,)]
        Beat positions in the requested units.
    """
    if onset_envelope is None:
        if y is None:
            from audiolib.exceptions import ParameterError
            raise ParameterError("one of y or onset_envelope must be provided")
        from audiolib.feature import onset_strength
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    oenv = onset_envelope.astype(np.float32).tolist()

    # Estimate tempo if not provided
    if bpm is None:
        bpm = float(_beat_tempo(oenv, float(sr), int(hop_length), float(start_bpm), 320.0))

    beat_frames = np.asarray(
        _beat_track_dp(oenv, float(bpm), float(sr), int(hop_length), float(tightness), trim),
        dtype=np.int32,
    )

    if units == "frames":
        beats_out = beat_frames
    elif units == "samples":
        beats_out = beat_frames * hop_length
    elif units == "time":
        beats_out = beat_frames * hop_length / sr
    else:
        from audiolib.exceptions import ParameterError
        raise ParameterError(f"Unknown units: {units!r}. Use 'frames', 'samples', or 'time'.")

    return np.float32(bpm), np.asarray(beats_out)
