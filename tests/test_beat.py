"""
Tests for audiolib.beat — beat tracking and tempo estimation.
"""
from __future__ import annotations

import numpy as np

from tests.conftest import HOP_LENGTH

# ---------------------------------------------------------------------------
# tempo
# ---------------------------------------------------------------------------

class TestTempo:
    def test_returns_scalar_or_array(self, sine_440, sr):
        from audiolib import beat
        t = beat.tempo(y=sine_440, sr=sr)
        assert np.ndim(t) <= 1

    def test_range_sane(self, sine_440, sr):
        from audiolib import beat
        t = beat.tempo(y=sine_440, sr=sr)
        val = float(np.atleast_1d(t)[0])
        assert 10.0 < val < 400.0

    def test_start_bpm_influence(self, sine_440, sr):
        from audiolib import beat
        # Providing a start_bpm should not crash
        t = beat.tempo(y=sine_440, sr=sr, start_bpm=120.0)
        val = float(np.atleast_1d(t)[0])
        assert val > 0

    def test_from_onset_envelope(self, sine_440, sr):
        from audiolib import beat
        from audiolib.feature import onset_strength
        oenv = onset_strength(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        t = beat.tempo(onset_envelope=oenv, sr=sr)
        val = float(np.atleast_1d(t)[0])
        assert val > 0

    def test_constant_120bpm_signal(self, sr):
        """A click-track at 120 BPM should be estimated near 120 BPM."""
        from audiolib import beat
        from audiolib.core import clicks
        bpm = 120.0
        # Generate click-track at 120 BPM for 4 seconds
        click_times = np.arange(0, 4.0, 60.0 / bpm)
        y = clicks(times=click_times.tolist(), sr=sr, length=int(4 * sr))
        t = beat.tempo(y=y, sr=sr, start_bpm=bpm)
        val = float(np.atleast_1d(t)[0])
        # Allow ±30% tolerance
        assert abs(val - bpm) / bpm < 0.3, f"Expected ~{bpm}, got {val}"


# ---------------------------------------------------------------------------
# beat_track
# ---------------------------------------------------------------------------

class TestBeatTrack:
    def test_returns_tuple(self, sine_440, sr):
        from audiolib import beat
        result = beat.beat_track(y=sine_440, sr=sr)
        assert len(result) == 2

    def test_tempo_positive(self, sine_440, sr):
        from audiolib import beat
        tempo, beats = beat.beat_track(y=sine_440, sr=sr)
        assert float(tempo) > 0.0

    def test_beat_frames_1d(self, sine_440, sr):
        from audiolib import beat
        tempo, beats = beat.beat_track(y=sine_440, sr=sr)
        beats = np.asarray(beats)
        assert beats.ndim == 1

    def test_beat_frames_increasing(self, sine_440, sr):
        from audiolib import beat
        tempo, beats = beat.beat_track(y=sine_440, sr=sr)
        beats = np.asarray(beats)
        if len(beats) > 1:
            assert (np.diff(beats) > 0).all()

    def test_from_onset_envelope(self, sine_440, sr):
        from audiolib import beat
        from audiolib.feature import onset_strength
        oenv = onset_strength(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        tempo, beats = beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH)
        assert float(tempo) > 0

    def test_units_time(self, sine_440, sr):
        from audiolib import beat
        tempo, beats = beat.beat_track(y=sine_440, sr=sr, units="time")
        beats = np.asarray(beats)
        # Should be in seconds
        if len(beats) > 0:
            assert beats[0] >= 0.0
            assert beats[-1] < len(sine_440) / sr + 1.0

    def test_units_samples(self, sine_440, sr):
        from audiolib import beat
        tempo, beats = beat.beat_track(y=sine_440, sr=sr, units="samples")
        beats = np.asarray(beats)
        if len(beats) > 0:
            assert beats[0] >= 0
            assert beats[-1] <= len(sine_440)

    def test_start_bpm(self, sine_440, sr):
        from audiolib import beat
        tempo, beats = beat.beat_track(y=sine_440, sr=sr, start_bpm=120.0)
        assert float(tempo) > 0

    def test_click_track_120bpm(self, sr):
        from audiolib import beat
        from audiolib.core import clicks
        bpm = 120.0
        click_times = np.arange(0, 4.0, 60.0 / bpm)
        y = clicks(times=click_times.tolist(), sr=sr, length=int(4 * sr))
        tempo, beats = beat.beat_track(y=y, sr=sr, start_bpm=bpm)
        val = float(np.atleast_1d(tempo)[0])
        assert abs(val - bpm) / bpm < 0.35, f"Expected ~{bpm}, got {val}"
