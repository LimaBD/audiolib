"""
Tests for audiolib.onset — onset detection functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import HOP_LENGTH, SR


# ---------------------------------------------------------------------------
# onset_detect
# ---------------------------------------------------------------------------

class TestOnsetDetect:
    def test_returns_1d_array(self, sine_440, sr):
        from audiolib import onset
        times = onset.onset_detect(y=sine_440, sr=sr)
        assert np.ndim(times) == 1

    def test_units_time_seconds(self, sine_440, sr):
        from audiolib import onset
        times = onset.onset_detect(y=sine_440, sr=sr, units="time")
        if len(times) > 0:
            assert times[0] >= 0.0
            assert times[-1] <= len(sine_440) / sr + 1.0

    def test_units_frames(self, sine_440, sr):
        from audiolib import onset
        frames = onset.onset_detect(y=sine_440, sr=sr, units="frames")
        if len(frames) > 0:
            assert frames[0] >= 0
            assert all(f >= 0 for f in frames)

    def test_units_samples(self, sine_440, sr):
        from audiolib import onset
        samples = onset.onset_detect(y=sine_440, sr=sr, units="samples")
        if len(samples) > 0:
            assert samples[0] >= 0
            assert samples[-1] <= len(sine_440)

    def test_from_onset_envelope(self, sine_440, sr):
        from audiolib import onset
        from audiolib.feature import onset_strength
        oenv = onset_strength(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        times = onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH)
        assert np.ndim(times) == 1

    def test_click_track_detects_clicks(self, sr):
        """Clicks at regular intervals should produce nearby onset detections."""
        from audiolib import onset
        from audiolib.core import clicks
        bpm = 120.0
        click_times = np.arange(0.1, 3.0, 60.0 / bpm).tolist()
        y = clicks(times=click_times, sr=sr, length=int(3 * sr))
        detected = onset.onset_detect(y=y, sr=sr, units="time")
        assert len(detected) > 0

    def test_silent_no_onsets(self, silent_signal, sr):
        from audiolib import onset
        times = onset.onset_detect(y=silent_signal, sr=sr)
        assert len(times) == 0

    def test_delta_param(self, sine_440, sr):
        from audiolib import onset
        # Higher threshold → fewer onsets
        times_strict = onset.onset_detect(y=sine_440, sr=sr, delta=0.5)
        times_loose = onset.onset_detect(y=sine_440, sr=sr, delta=0.001)
        assert len(times_loose) >= len(times_strict)

    def test_wait_param(self, sr):
        from audiolib import onset
        from audiolib.core import clicks
        # Clicks at 120 bpm with different wait constraints
        click_times = np.arange(0.1, 3.0, 0.5).tolist()
        y = clicks(times=click_times, sr=sr, length=int(3 * sr))
        times_1 = onset.onset_detect(y=y, sr=sr, wait=1)
        times_long = onset.onset_detect(y=y, sr=sr, wait=100)
        assert len(times_long) <= len(times_1)

    def test_monotone_increasing(self, sine_440, sr):
        from audiolib import onset
        from audiolib.core import clicks
        click_times = np.arange(0.1, 2.0, 0.25).tolist()
        y = clicks(times=click_times, sr=sr, length=int(2 * sr))
        times = onset.onset_detect(y=y, sr=sr, units="time")
        if len(times) > 1:
            assert (np.diff(times) > 0).all()
