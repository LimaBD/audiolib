"""
Compatibility tests: numerically cross-check audiolib against librosa.

Run with the real librosa installed:
    LIBROSA_COMPAT=1 pytest tests/test_compat.py -v

These tests are skipped automatically when librosa is not installed or
when LIBROSA_COMPAT env var is not set (so CI doesn't require librosa).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

LIBROSA_COMPAT = os.environ.get("LIBROSA_COMPAT", "0") == "1"
librosa = pytest.importorskip("librosa") if LIBROSA_COMPAT else None

pytestmark = pytest.mark.skipif(
    not LIBROSA_COMPAT,
    reason="Set LIBROSA_COMPAT=1 and install librosa to run compatibility tests",
)

# Tolerances — float32 DSP is inherently looser than float64
ATOL = 5e-3
RTOL = 1e-2


# ---------------------------------------------------------------------------
# Fixtures (local overrides to avoid session-scope issues)
# ---------------------------------------------------------------------------

@pytest.fixture
def sr():
    return 22050


@pytest.fixture
def y(sr):
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class TestCompatSTFT:
    def test_stft_magnitude(self, y):
        import audiolib.core as lx
        D_lx = np.abs(lx.stft(y, n_fft=2048, hop_length=512))
        D_lr = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        np.testing.assert_allclose(D_lx, D_lr, atol=ATOL, rtol=RTOL)

    def test_istft_reconstruction(self, y):
        import audiolib.core as lx
        D = lx.stft(y, n_fft=2048, hop_length=512)
        y_back = lx.istft(D, hop_length=512, length=len(y))
        y_lr = librosa.istft(librosa.stft(y, n_fft=2048, hop_length=512), hop_length=512, length=len(y))
        np.testing.assert_allclose(y_back, y_lr, atol=ATOL)


class TestCompatMagnitudeScaling:
    def test_amplitude_to_db(self, y):
        import audiolib.core as lx
        mag = np.abs(lx.stft(y))
        db_lx = lx.amplitude_to_db(mag)
        db_lr = librosa.amplitude_to_db(mag)
        np.testing.assert_allclose(db_lx, db_lr, atol=ATOL)

    def test_power_to_db(self, y):
        import audiolib.core as lx
        mag = np.abs(lx.stft(y)) ** 2
        db_lx = lx.power_to_db(mag)
        db_lr = librosa.power_to_db(mag)
        np.testing.assert_allclose(db_lx, db_lr, atol=ATOL)


class TestCompatZeroCrossings:
    def test_zero_crossings(self, y):
        import audiolib.core as lx
        zc_lx = lx.zero_crossings(y)
        zc_lr = librosa.zero_crossings(y)
        # Allow small differences at edges
        np.testing.assert_array_equal(zc_lx[1:-1], zc_lr[1:-1])


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

class TestCompatMelspectrogram:
    def test_melspectrogram(self, y, sr):
        import audiolib.feature as lxf
        M_lx = lxf.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        M_lr = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        np.testing.assert_allclose(M_lx, M_lr, atol=ATOL * 10, rtol=RTOL)


class TestCompatMFCC:
    def test_mfcc_shape(self, y, sr):
        import audiolib.feature as lxf
        mfcc_lx = lxf.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_lr = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        assert mfcc_lx.shape == mfcc_lr.shape

    def test_mfcc_first_coeff(self, y, sr):
        """First MFCC (energy) should be roughly correlated."""
        import audiolib.feature as lxf
        mfcc_lx = lxf.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_lr = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        corr = np.corrcoef(mfcc_lx[0], mfcc_lr[0])[0, 1]
        assert corr > 0.95, f"First MFCC correlation too low: {corr}"


class TestCompatChromaSTFT:
    def test_chroma_shape(self, y, sr):
        import audiolib.feature as lxf
        ch_lx = lxf.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        ch_lr = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        assert ch_lx.shape == ch_lr.shape

    def test_chroma_peak_bin(self, y, sr):
        """Peak chroma bin for A440 should agree between implementations."""
        import audiolib.feature as lxf
        ch_lx = lxf.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=512)
        ch_lr = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=512)
        peak_lx = np.argmax(ch_lx.mean(axis=1))
        peak_lr = np.argmax(ch_lr.mean(axis=1))
        assert peak_lx == peak_lr


class TestCompatSpectralFeatures:
    def test_spectral_centroid(self, y, sr):
        import audiolib.feature as lxf
        sc_lx = lxf.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
        sc_lr = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
        np.testing.assert_allclose(sc_lx, sc_lr, atol=50.0)  # within 50 Hz

    def test_spectral_rolloff(self, y, sr):
        import audiolib.feature as lxf
        sr_lx = lxf.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=512)
        sr_lr = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=512)
        np.testing.assert_allclose(sr_lx, sr_lr, rtol=0.1)


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

class TestCompatConvert:
    @pytest.mark.parametrize("hz", [0.0, 110.0, 440.0, 1000.0, 8000.0])
    def test_hz_to_mel_slaney(self, hz):
        from audiolib.convert import hz_to_mel
        lx_mel = hz_to_mel(hz, htk=False)
        lr_mel = librosa.hz_to_mel(hz, htk=False)
        assert abs(lx_mel - lr_mel) < 0.01

    @pytest.mark.parametrize("hz", [110.0, 440.0, 1000.0, 8000.0])
    def test_hz_to_mel_htk(self, hz):
        from audiolib.convert import hz_to_mel
        lx_mel = hz_to_mel(hz, htk=True)
        lr_mel = librosa.hz_to_mel(hz, htk=True)
        assert abs(lx_mel - lr_mel) < 0.01

    def test_fft_frequencies(self):
        from audiolib.convert import fft_frequencies
        lx = fft_frequencies(sr=22050, n_fft=2048)
        lr = librosa.fft_frequencies(sr=22050, n_fft=2048)
        np.testing.assert_allclose(lx, lr, atol=1e-3)

    def test_mel_frequencies(self):
        from audiolib.convert import mel_frequencies
        lx = mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0, htk=False)
        lr = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0, htk=False)
        np.testing.assert_allclose(lx, lr, atol=0.1)

    def test_frames_to_time(self):
        from audiolib.convert import frames_to_time
        frames = np.arange(10)
        lx = frames_to_time(frames, hop_length=512, sr=22050)
        lr = librosa.frames_to_time(frames, hop_length=512, sr=22050)
        np.testing.assert_allclose(lx, lr, atol=1e-6)

    @pytest.mark.parametrize("note", ["C4", "A4", "G#3", "Bb5"])
    def test_note_to_midi(self, note):
        from audiolib.convert import note_to_midi
        lx = note_to_midi(note)
        lr = librosa.note_to_midi(note)
        assert lx == lr
