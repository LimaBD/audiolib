"""
Tests for audiolib.core — audio I/O and core DSP functions.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import HOP_LENGTH, N_FFT, N_SAMPLES

# ---------------------------------------------------------------------------
# to_mono
# ---------------------------------------------------------------------------

class TestToMono:
    def test_already_mono(self, sine_440):
        from audiolib.core import to_mono
        out = to_mono(sine_440)
        np.testing.assert_array_equal(out, sine_440)

    def test_stereo_averages_channels(self, stereo_sine):
        from audiolib.core import to_mono
        out = to_mono(stereo_sine)
        assert out.ndim == 1
        assert out.shape[0] == stereo_sine.shape[1]

    def test_stereo_dtype_preserved(self, stereo_sine):
        from audiolib.core import to_mono
        out = to_mono(stereo_sine)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------

class TestResample:
    def test_downsample_length(self, sine_440, sr):
        from audiolib.core import resample
        target_sr = sr // 2
        out = resample(sine_440, orig_sr=sr, target_sr=target_sr)
        expected_len = int(len(sine_440) * target_sr / sr)
        assert abs(len(out) - expected_len) <= 2

    def test_upsample_length(self, sine_440, sr):
        from audiolib.core import resample
        target_sr = sr * 2
        out = resample(sine_440, orig_sr=sr, target_sr=target_sr)
        expected_len = int(len(sine_440) * target_sr / sr)
        assert abs(len(out) - expected_len) <= 2

    def test_same_sr_noop(self, sine_440, sr):
        from audiolib.core import resample
        out = resample(sine_440, orig_sr=sr, target_sr=sr)
        assert len(out) == len(sine_440)

    def test_output_dtype_float32(self, sine_440, sr):
        from audiolib.core import resample
        out = resample(sine_440, orig_sr=sr, target_sr=sr // 2)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# get_duration / get_samplerate
# ---------------------------------------------------------------------------

class TestGetDuration:
    def test_from_signal(self, sine_440, sr):
        from audiolib.core import get_duration
        dur = get_duration(y=sine_440, sr=sr)
        assert abs(dur - 1.0) < 0.01

    def test_from_file(self, sine_440, sr):
        from audiolib.core import get_duration
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, sine_440, sr)
            dur = get_duration(path=path)
            assert abs(dur - 1.0) < 0.02
        finally:
            os.unlink(path)

    def test_from_stft(self, stft_matrix, sr):
        from audiolib.core import get_duration
        dur = get_duration(S=stft_matrix, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
        assert dur > 0.0


class TestGetSamplerate:
    def test_reads_samplerate(self, sine_440, sr):
        from audiolib.core import get_samplerate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, sine_440, sr)
            assert get_samplerate(path) == sr
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    def _write_wav(self, y, sr):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
        return f.name

    def test_load_mono(self, sine_440, sr):
        from audiolib.core import load
        path = self._write_wav(sine_440, sr)
        try:
            y, loaded_sr = load(path, sr=None)
            assert loaded_sr == sr
            assert y.ndim == 1
            assert y.dtype == np.float32
        finally:
            os.unlink(path)

    def test_load_with_resample(self, sine_440, sr):
        from audiolib.core import load
        path = self._write_wav(sine_440, sr)
        try:
            y, loaded_sr = load(path, sr=sr // 2)
            assert loaded_sr == sr // 2
            expected = int(N_SAMPLES * (sr // 2) / sr)
            assert abs(len(y) - expected) <= 2
        finally:
            os.unlink(path)

    def test_load_mono_from_stereo(self, stereo_sine, sr):
        from audiolib.core import load
        path = self._write_wav(stereo_sine.T, sr)  # soundfile expects (samples, channels)
        try:
            y, loaded_sr = load(path, sr=None, mono=True)
            assert y.ndim == 1
        finally:
            os.unlink(path)

    def test_load_duration_offset(self, sine_440, sr):
        from audiolib.core import load
        path = self._write_wav(sine_440, sr)
        try:
            y, loaded_sr = load(path, sr=None, offset=0.1, duration=0.5)
            expected = int(0.5 * sr)
            assert abs(len(y) - expected) <= sr // 100
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# STFT / ISTFT
# ---------------------------------------------------------------------------

class TestSTFT:
    def test_output_shape(self, sine_440):
        from audiolib.core import stft
        D = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH)
        n_bins = N_FFT // 2 + 1
        assert D.shape[0] == n_bins
        assert D.ndim == 2

    def test_output_dtype_complex(self, sine_440):
        from audiolib.core import stft
        D = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH)
        assert np.iscomplexobj(D)

    def test_custom_win_length(self, sine_440):
        from audiolib.core import stft
        D = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT // 2)
        assert D.shape[0] == N_FFT // 2 + 1

    def test_center_padding(self, sine_440):
        from audiolib.core import stft
        D_center = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
        D_nocenter = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False)
        assert D_center.shape[1] >= D_nocenter.shape[1]


class TestISTFT:
    def test_roundtrip_shape(self, sine_440):
        from audiolib.core import istft, stft
        D = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH)
        y_back = istft(D, hop_length=HOP_LENGTH, length=len(sine_440))
        assert len(y_back) == len(sine_440)

    def test_roundtrip_fidelity(self, sine_440):
        from audiolib.core import istft, stft
        D = stft(sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH)
        y_back = istft(D, hop_length=HOP_LENGTH, length=len(sine_440))
        # Exclude boundary N_FFT//2 samples on each side (edge effects from zero-padding)
        trim = N_FFT // 2
        np.testing.assert_allclose(y_back[trim:-trim], sine_440[trim:-trim], atol=1e-3)


# ---------------------------------------------------------------------------
# magphase
# ---------------------------------------------------------------------------

class TestMagphase:
    def test_magnitude_positive(self, stft_matrix):
        from audiolib.core import magphase
        mag, phase = magphase(stft_matrix)
        assert (mag >= 0).all()

    def test_phase_unit_circle(self, stft_matrix):
        from audiolib.core import magphase
        mag, phase = magphase(stft_matrix)
        np.testing.assert_allclose(np.abs(phase), 1.0, atol=1e-6)

    def test_reconstruction(self, stft_matrix):
        from audiolib.core import magphase
        mag, phase = magphase(stft_matrix)
        np.testing.assert_allclose(np.abs(mag * phase - stft_matrix), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# amplitude_to_db / power_to_db / db_to_amplitude / db_to_power
# ---------------------------------------------------------------------------

class TestMagnitudeScaling:
    def test_amplitude_to_db_shape(self, stft_matrix):
        from audiolib.core import amplitude_to_db
        mag = np.abs(stft_matrix)
        db = amplitude_to_db(mag)
        assert db.shape == mag.shape

    def test_power_to_db_shape(self, stft_matrix):
        from audiolib.core import power_to_db
        mag = np.abs(stft_matrix) ** 2
        db = power_to_db(mag)
        assert db.shape == mag.shape

    def test_db_roundtrip_amplitude(self):
        from audiolib.core import amplitude_to_db, db_to_amplitude
        x = np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float32)
        db = amplitude_to_db(x)
        x_back = db_to_amplitude(db)
        np.testing.assert_allclose(x_back, x, rtol=1e-4)

    def test_db_roundtrip_power(self):
        from audiolib.core import db_to_power, power_to_db
        x = np.array([1e-4, 1e-3, 1.0, 1e3], dtype=np.float32)
        db = power_to_db(x)
        x_back = db_to_power(db)
        np.testing.assert_allclose(x_back, x, rtol=1e-4)

    def test_amplitude_to_db_range(self, stft_matrix):
        from audiolib.core import amplitude_to_db
        mag = np.abs(stft_matrix)
        db = amplitude_to_db(mag, top_db=80.0)
        assert (db >= db.max() - 80.0 - 1e-4).all()


# ---------------------------------------------------------------------------
# zero_crossings
# ---------------------------------------------------------------------------

class TestZeroCrossings:
    def test_basic_shape(self, sine_440):
        from audiolib.core import zero_crossings
        zc = zero_crossings(sine_440)
        assert zc.shape == sine_440.shape
        assert zc.dtype == bool

    def test_sine_has_crossings(self, sine_440):
        from audiolib.core import zero_crossings
        zc = zero_crossings(sine_440)
        assert zc.sum() > 0

    def test_constant_no_crossings(self, sr):
        from audiolib.core import zero_crossings
        y = np.ones(sr, dtype=np.float32)
        zc = zero_crossings(y)
        assert zc.sum() == 0


# ---------------------------------------------------------------------------
# autocorrelate
# ---------------------------------------------------------------------------

class TestAutocorrelate:
    def test_max_at_zero_lag(self, sine_440):
        from audiolib.core import autocorrelate
        ac = autocorrelate(sine_440)
        assert ac[0] == pytest.approx(ac.max(), rel=1e-3)

    def test_output_length_default(self, sine_440):
        from audiolib.core import autocorrelate
        ac = autocorrelate(sine_440)
        assert len(ac) == len(sine_440)

    def test_output_length_max_size(self, sine_440):
        from audiolib.core import autocorrelate
        ac = autocorrelate(sine_440, max_size=100)
        assert len(ac) == 100


# ---------------------------------------------------------------------------
# mu_compress / mu_expand
# ---------------------------------------------------------------------------

class TestMuLaw:
    def test_compress_expand_roundtrip(self, sine_440):
        from audiolib.core import mu_compress, mu_expand
        compressed = mu_compress(sine_440)
        expanded = mu_expand(compressed)
        np.testing.assert_allclose(expanded, sine_440, atol=1e-4)

    def test_compress_output_range(self, sine_440):
        from audiolib.core import mu_compress
        compressed = mu_compress(sine_440, quantize=False)
        # Output should lie in [-1, 1]
        assert compressed.min() >= -1.0 - 1e-6
        assert compressed.max() <= 1.0 + 1e-6

    def test_quantize_integer_output(self, sine_440):
        from audiolib.core import mu_compress
        quantized = mu_compress(sine_440, quantize=True)
        # Quantized values should be integers (stored as float)
        assert np.all(quantized == np.round(quantized))


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

class TestTone:
    def test_output_length(self, sr):
        from audiolib.core import tone
        y = tone(440.0, sr=sr, duration=0.5)
        assert len(y) == int(0.5 * sr)

    def test_output_dtype(self, sr):
        from audiolib.core import tone
        y = tone(440.0, sr=sr, duration=1.0)
        assert y.dtype == np.float32


class TestChirp:
    def test_output_length(self, sr):
        pytest.importorskip("scipy")
        from audiolib.core import chirp
        y = chirp(fmin=220.0, fmax=880.0, sr=sr, duration=1.0)
        assert len(y) == sr

    def test_output_dtype(self, sr):
        pytest.importorskip("scipy")
        from audiolib.core import chirp
        y = chirp(fmin=220.0, fmax=880.0, sr=sr, duration=1.0)
        assert y.dtype == np.float32


class TestClicks:
    def test_default_output(self, sr):
        from audiolib.core import clicks
        y = clicks(times=[0.1, 0.5, 0.9], sr=sr)
        assert y.ndim == 1
        assert len(y) > 0
