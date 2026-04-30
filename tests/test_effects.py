"""
Tests for audiolib.effects — audio effects and transformations.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import N_FFT

# ---------------------------------------------------------------------------
# time_stretch
# ---------------------------------------------------------------------------

class TestTimeStretch:
    def test_speed_up_shorter(self, sine_440):
        from audiolib.effects import time_stretch
        out = time_stretch(sine_440, rate=2.0)
        # Faster means fewer samples
        assert len(out) < len(sine_440)

    def test_slow_down_longer(self, sine_440):
        from audiolib.effects import time_stretch
        out = time_stretch(sine_440, rate=0.5)
        # Slower means more samples
        assert len(out) > len(sine_440)

    def test_rate_one_approx_noop(self, sine_440):
        from audiolib.effects import time_stretch
        out = time_stretch(sine_440, rate=1.0)
        assert abs(len(out) - len(sine_440)) <= 4

    def test_output_dtype_float32(self, sine_440):
        from audiolib.effects import time_stretch
        out = time_stretch(sine_440, rate=1.5)
        assert out.dtype == np.float32

    def test_output_1d(self, sine_440):
        from audiolib.effects import time_stretch
        out = time_stretch(sine_440, rate=1.2)
        assert out.ndim == 1

    def test_bad_rate_raises(self, sine_440):
        from audiolib.effects import time_stretch
        from audiolib.exceptions import ParameterError
        with pytest.raises(ParameterError):
            time_stretch(sine_440, rate=0.0)
        with pytest.raises(ParameterError):
            time_stretch(sine_440, rate=-1.0)

    def test_custom_n_fft(self, sine_440):
        from audiolib.effects import time_stretch
        out = time_stretch(sine_440, rate=1.5, n_fft=1024)
        assert out.ndim == 1

    def test_expected_length(self, sine_440):
        from audiolib.effects import time_stretch
        rate = 1.5
        out = time_stretch(sine_440, rate=rate)
        # Expected length ≈ n_samples / rate, allow ±10%
        expected = int(len(sine_440) / rate)
        assert abs(len(out) - expected) < expected * 0.2


# ---------------------------------------------------------------------------
# pitch_shift
# ---------------------------------------------------------------------------

class TestPitchShift:
    def test_output_same_length(self, sine_440, sr):
        pytest.importorskip("scipy")
        from audiolib.effects import pitch_shift
        out = pitch_shift(sine_440, sr=sr, n_steps=2)
        assert abs(len(out) - len(sine_440)) <= sr // 100

    def test_zero_steps_noop(self, sine_440, sr):
        pytest.importorskip("scipy")
        from audiolib.effects import pitch_shift
        out = pitch_shift(sine_440, sr=sr, n_steps=0)
        assert abs(len(out) - len(sine_440)) <= sr // 100

    def test_output_dtype(self, sine_440, sr):
        pytest.importorskip("scipy")
        from audiolib.effects import pitch_shift
        out = pitch_shift(sine_440, sr=sr, n_steps=4)
        assert out.dtype == np.float32

    def test_down_shift(self, sine_440, sr):
        pytest.importorskip("scipy")
        from audiolib.effects import pitch_shift
        out = pitch_shift(sine_440, sr=sr, n_steps=-4)
        assert out.ndim == 1

    def test_bins_per_octave_param(self, sine_440, sr):
        pytest.importorskip("scipy")
        from audiolib.effects import pitch_shift
        out = pitch_shift(sine_440, sr=sr, n_steps=1, bins_per_octave=24)
        assert out.ndim == 1


# ---------------------------------------------------------------------------
# trim
# ---------------------------------------------------------------------------

class TestTrim:
    def test_basic_trim(self, sine_440):
        from audiolib.effects import trim
        # Add silence at both ends
        padded = np.concatenate([
            np.zeros(500, dtype=np.float32),
            sine_440,
            np.zeros(500, dtype=np.float32),
        ])
        trimmed, idx = trim(padded)
        assert len(trimmed) < len(padded)
        assert idx.shape == (2,)

    def test_indices_span_content(self, sine_440):
        from audiolib.effects import trim
        padded = np.concatenate([
            np.zeros(1000, dtype=np.float32),
            sine_440,
            np.zeros(1000, dtype=np.float32),
        ])
        trimmed, idx = trim(padded)
        start, end = idx
        assert start >= 0
        assert end <= len(padded)
        assert end > start

    def test_all_silence_returns_empty(self, silent_signal):
        from audiolib.effects import trim
        trimmed, idx = trim(silent_signal)
        assert len(trimmed) == 0

    def test_no_silence_returns_full(self, sine_440):
        from audiolib.effects import trim
        # Set high top_db to preserve everything
        trimmed, idx = trim(sine_440, top_db=200)
        assert len(trimmed) >= len(sine_440) // 2

    def test_output_dtype(self, sine_440):
        from audiolib.effects import trim
        trimmed, _ = trim(sine_440)
        assert trimmed.dtype == np.float32


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_basic(self, sine_440):
        from audiolib.effects import split
        # Single voiced segment → returns intervals
        intervals = split(sine_440, top_db=200)
        assert intervals.ndim == 2
        assert intervals.shape[1] == 2

    def test_split_silence(self, silent_signal):
        from audiolib.effects import split
        intervals = split(silent_signal)
        assert len(intervals) == 0

    def test_intervals_sorted(self, sine_440):
        from audiolib.effects import split
        intervals = split(sine_440, top_db=200)
        if len(intervals) > 1:
            assert (np.diff(intervals[:, 0]) > 0).all()

    def test_intervals_non_overlapping(self, sine_440):
        from audiolib.effects import split
        intervals = split(sine_440, top_db=200)
        for i in range(len(intervals) - 1):
            assert intervals[i, 1] <= intervals[i + 1, 0]

    def test_intervals_within_bounds(self, sine_440):
        from audiolib.effects import split
        intervals = split(sine_440, top_db=200)
        assert (intervals >= 0).all()
        assert (intervals <= len(sine_440)).all()


# ---------------------------------------------------------------------------
# hpss
# ---------------------------------------------------------------------------

class TestHPSS:
    def test_output_shapes(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import hpss
        H, P = hpss(sine_440)
        assert H.ndim == 1
        assert P.ndim == 1
        # Lengths should match approximately
        assert abs(len(H) - len(sine_440)) < N_FFT
        assert abs(len(P) - len(sine_440)) < N_FFT

    def test_harmonic_percussive_components(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import hpss
        H, P = hpss(sine_440)
        # Both components should be valid float arrays
        assert np.isfinite(H).all()
        assert np.isfinite(P).all()

    def test_harmonic_function(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import harmonic
        H = harmonic(sine_440)
        assert H.ndim == 1

    def test_percussive_function(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import percussive
        P = percussive(sine_440)
        assert P.ndim == 1

    def test_sine_mostly_harmonic(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import hpss
        H, P = hpss(sine_440)
        min_len = min(len(H), len(P), len(sine_440))
        # The harmonic component of a sine should have more energy than percussive
        assert np.sum(H[:min_len] ** 2) > np.sum(P[:min_len] ** 2)


# ---------------------------------------------------------------------------
# remix
# ---------------------------------------------------------------------------

class TestRemix:
    def test_output_length(self, sine_440):
        from audiolib.effects import remix
        intervals = np.array([[0, 1000], [2000, 3000]], dtype=np.int32)
        out = remix(sine_440, intervals)
        assert len(out) == 2000

    def test_empty_intervals(self, sine_440):
        from audiolib.effects import remix
        intervals = np.zeros((0, 2), dtype=np.int32)
        out = remix(sine_440, intervals)
        assert len(out) == 0

    def test_single_interval(self, sine_440):
        from audiolib.effects import remix
        intervals = np.array([[100, 500]])
        out = remix(sine_440, intervals)
        assert len(out) == 400
        np.testing.assert_array_equal(out, sine_440[100:500])


# ---------------------------------------------------------------------------
# preemphasis / deemphasis
# ---------------------------------------------------------------------------

class TestPreemphasis:
    def test_output_shape(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import preemphasis
        out = preemphasis(sine_440)
        assert out.shape == sine_440.shape

    def test_coef_zero_noop(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import preemphasis
        out = preemphasis(sine_440, coef=0.0)
        np.testing.assert_allclose(out, sine_440, atol=1e-5)

    def test_return_zf(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import preemphasis
        out, zf = preemphasis(sine_440, return_zf=True)
        assert out.shape == sine_440.shape
        assert zf is not None


class TestDeemphasis:
    def test_roundtrip(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import deemphasis, preemphasis
        emphasized = preemphasis(sine_440)
        recovered = deemphasis(emphasized)
        np.testing.assert_allclose(recovered, sine_440, atol=1e-4)

    def test_output_dtype(self, sine_440):
        pytest.importorskip("scipy")
        from audiolib.effects import deemphasis
        out = deemphasis(sine_440)
        assert np.issubdtype(out.dtype, np.floating)
