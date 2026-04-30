"""
Tests for audiolib.pitch — pitch estimation functions.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# yin
# ---------------------------------------------------------------------------

class TestYIN:
    def test_output_1d(self, sine_440, sr):
        from audiolib import pitch
        f0 = pitch.yin(sine_440, fmin=50.0, fmax=2000.0, sr=sr)
        assert np.ndim(f0) == 1
        assert len(f0) > 0

    def test_output_dtype_float(self, sine_440, sr):
        from audiolib import pitch
        f0 = pitch.yin(sine_440, fmin=50.0, fmax=2000.0, sr=sr)
        assert np.issubdtype(np.array(f0).dtype, np.floating)

    def test_voiced_near_440(self, sine_440, sr):
        from audiolib import pitch
        f0 = pitch.yin(sine_440, fmin=100.0, fmax=1000.0, sr=sr)
        f0_arr = np.array(f0)
        voiced = f0_arr[f0_arr > 0]
        assert len(voiced) > 0
        # Median voiced estimate should be near 440 Hz (within ±20%)
        assert abs(np.median(voiced) - 440.0) / 440.0 < 0.20, (
            f"Expected ~440 Hz, got {np.median(voiced):.1f} Hz"
        )

    def test_silent_mostly_unvoiced(self, silent_signal, sr):
        from audiolib import pitch
        f0 = pitch.yin(silent_signal, fmin=50.0, fmax=2000.0, sr=sr)
        f0_arr = np.array(f0)
        unvoiced_ratio = (f0_arr == 0.0).mean()
        assert unvoiced_ratio > 0.9

    def test_fmin_fmax_constraint(self, sine_440, sr):
        from audiolib import pitch
        fmin, fmax = 200.0, 800.0
        f0 = pitch.yin(sine_440, fmin=fmin, fmax=fmax, sr=sr)
        f0_arr = np.array(f0)
        voiced = f0_arr[f0_arr > 0]
        # All voiced estimates should be within [fmin, fmax]
        if len(voiced) > 0:
            assert voiced.min() >= fmin * 0.8
            assert voiced.max() <= fmax * 1.2

    def test_hop_length_affects_frames(self, sine_440, sr):
        from audiolib import pitch
        f0_short = pitch.yin(sine_440, fmin=50.0, fmax=2000.0, sr=sr, hop_length=256)
        f0_long = pitch.yin(sine_440, fmin=50.0, fmax=2000.0, sr=sr, hop_length=1024)
        assert len(f0_short) > len(f0_long)

    def test_bad_fmin_raises(self, sine_440, sr):
        from audiolib import pitch
        from audiolib.exceptions import ParameterError
        with pytest.raises(ParameterError):
            pitch.yin(sine_440, fmin=0.0, fmax=2000.0, sr=sr)

    def test_fmin_greater_fmax_raises(self, sine_440, sr):
        from audiolib import pitch
        from audiolib.exceptions import ParameterError
        with pytest.raises(ParameterError):
            pitch.yin(sine_440, fmin=1000.0, fmax=500.0, sr=sr)

    def test_frame_length_param(self, sine_440, sr):
        from audiolib import pitch
        f0 = pitch.yin(sine_440, fmin=50.0, fmax=2000.0, sr=sr, frame_length=1024)
        assert len(f0) > 0

    def test_trough_threshold_param(self, sine_440, sr):
        from audiolib import pitch
        f0 = pitch.yin(sine_440, fmin=50.0, fmax=2000.0, sr=sr, trough_threshold=0.2)
        assert len(f0) > 0
