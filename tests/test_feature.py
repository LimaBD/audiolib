"""
Tests for audiolib.feature — feature extraction functions.
"""
from __future__ import annotations

import numpy as np

from tests.conftest import HOP_LENGTH, N_FFT, N_SAMPLES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def n_frames_for(n_samples=N_SAMPLES, hop_length=HOP_LENGTH, n_fft=N_FFT, center=True):
    """Expected number of STFT frames."""
    n = n_samples + n_fft if center else n_samples
    return 1 + (n - n_fft) // hop_length


# ---------------------------------------------------------------------------
# melspectrogram
# ---------------------------------------------------------------------------

class TestMelspectrogram:
    def test_shape_from_audio(self, sine_440, sr):
        from audiolib import feature
        M = feature.melspectrogram(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        n_mels = 128
        nf = n_frames_for(len(sine_440))
        assert M.shape == (n_mels, nf), f"Expected ({n_mels}, {nf}), got {M.shape}"

    def test_shape_from_stft(self, stft_matrix, sr):
        from audiolib import feature
        n_bins, nf = stft_matrix.shape
        M = feature.melspectrogram(S=np.abs(stft_matrix) ** 2, sr=sr, n_fft=N_FFT)
        assert M.shape[1] == nf
        assert M.shape[0] == 128

    def test_non_negative(self, sine_440, sr):
        from audiolib import feature
        M = feature.melspectrogram(y=sine_440, sr=sr)
        assert (M >= 0).all()

    def test_n_mels_param(self, sine_440, sr):
        from audiolib import feature
        M = feature.melspectrogram(y=sine_440, sr=sr, n_mels=64)
        assert M.shape[0] == 64

    def test_fmin_fmax(self, sine_440, sr):
        from audiolib import feature
        M = feature.melspectrogram(y=sine_440, sr=sr, fmin=80.0, fmax=8000.0)
        assert M.shape[0] == 128

    def test_htk_scale(self, sine_440, sr):
        from audiolib import feature
        M = feature.melspectrogram(y=sine_440, sr=sr, htk=True)
        assert M.shape[0] == 128


# ---------------------------------------------------------------------------
# mfcc
# ---------------------------------------------------------------------------

class TestMFCC:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        nf = n_frames_for(len(sine_440))
        assert mfcc.shape == (13, nf)

    def test_n_mfcc_param(self, sine_440, sr):
        from audiolib import feature
        for n in [13, 20, 40]:
            mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=n)
            assert mfcc.shape[0] == n

    def test_from_S(self, stft_matrix, sr):
        from audiolib import feature
        mfcc = feature.mfcc(S=np.abs(stft_matrix) ** 2, sr=sr, n_mfcc=13)
        assert mfcc.shape[0] == 13

    def test_dtype_float(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr)
        assert np.issubdtype(mfcc.dtype, np.floating)


# ---------------------------------------------------------------------------
# chroma_stft
# ---------------------------------------------------------------------------

class TestChromaSTFT:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        chroma = feature.chroma_stft(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert chroma.shape == (12, nf)

    def test_values_in_0_1(self, sine_440, sr):
        from audiolib import feature
        chroma = feature.chroma_stft(y=sine_440, sr=sr)
        assert chroma.min() >= -1e-6
        assert chroma.max() <= 1.0 + 1e-6

    def test_440hz_peak_on_A(self, sine_440, sr):
        from audiolib import feature
        # 440 Hz = A4; chroma bin 9 = A
        chroma = feature.chroma_stft(y=sine_440, sr=sr, n_fft=4096, hop_length=HOP_LENGTH)
        mean_chroma = chroma.mean(axis=1)
        peak = np.argmax(mean_chroma)
        assert peak == 9, f"Expected peak at bin 9 (A), got {peak}"


# ---------------------------------------------------------------------------
# spectral_centroid
# ---------------------------------------------------------------------------

class TestSpectralCentroid:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        sc = feature.spectral_centroid(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert sc.shape == (1, nf)

    def test_positive(self, sine_440, sr):
        from audiolib import feature
        sc = feature.spectral_centroid(y=sine_440, sr=sr)
        assert (sc >= 0).all()

    def test_440hz_centroid_near_440(self, sine_440, sr):
        from audiolib import feature
        sc = feature.spectral_centroid(y=sine_440, sr=sr, n_fft=4096, hop_length=HOP_LENGTH)
        # Most frames should have centroid close to 440 Hz
        assert abs(np.median(sc) - 440.0) < 100.0


# ---------------------------------------------------------------------------
# spectral_bandwidth
# ---------------------------------------------------------------------------

class TestSpectralBandwidth:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        sb = feature.spectral_bandwidth(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert sb.shape == (1, nf)

    def test_non_negative(self, sine_440, sr):
        from audiolib import feature
        sb = feature.spectral_bandwidth(y=sine_440, sr=sr)
        assert (sb >= 0).all()


# ---------------------------------------------------------------------------
# spectral_rolloff
# ---------------------------------------------------------------------------

class TestSpectralRolloff:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        sr_feat = feature.spectral_rolloff(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert sr_feat.shape == (1, nf)

    def test_range(self, sine_440, sr):
        from audiolib import feature
        sr_feat = feature.spectral_rolloff(y=sine_440, sr=sr)
        # Rolloff frequency must be between 0 and Nyquist
        assert (sr_feat >= 0).all()
        assert (sr_feat <= sr / 2).all()


# ---------------------------------------------------------------------------
# spectral_flatness
# ---------------------------------------------------------------------------

class TestSpectralFlatness:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        sf_feat = feature.spectral_flatness(y=sine_440, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert sf_feat.shape == (1, nf)

    def test_sine_low_flatness(self, sine_440):
        from audiolib import feature
        # A pure tone has low flatness (concentrated spectrum)
        sf_feat = feature.spectral_flatness(y=sine_440, n_fft=4096)
        assert sf_feat.mean() < 0.5

    def test_noise_high_flatness(self, white_noise):
        from audiolib import feature
        # White noise has high flatness (flat spectrum)
        sf_feat = feature.spectral_flatness(y=white_noise, n_fft=4096)
        assert sf_feat.mean() > 0.3


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

class TestRMS:
    def test_shape(self, sine_440):
        from audiolib import feature
        r = feature.rms(y=sine_440, frame_length=N_FFT, hop_length=HOP_LENGTH)
        assert r.shape[0] == 1

    def test_silent_is_zero(self, silent_signal):
        from audiolib import feature
        r = feature.rms(y=silent_signal)
        np.testing.assert_allclose(r, 0.0, atol=1e-6)

    def test_from_S(self, stft_matrix):
        from audiolib import feature
        r = feature.rms(S=np.abs(stft_matrix))
        assert r.shape[0] == 1
        assert (r >= 0).all()


# ---------------------------------------------------------------------------
# zero_crossing_rate
# ---------------------------------------------------------------------------

class TestZeroCrossingRate:
    def test_shape(self, sine_440):
        from audiolib import feature
        zcr = feature.zero_crossing_rate(sine_440, frame_length=N_FFT, hop_length=HOP_LENGTH)
        assert zcr.shape[0] == 1

    def test_silent_is_zero(self, silent_signal):
        from audiolib import feature
        zcr = feature.zero_crossing_rate(silent_signal)
        np.testing.assert_allclose(zcr, 0.0, atol=1e-6)

    def test_range(self, sine_440):
        from audiolib import feature
        zcr = feature.zero_crossing_rate(sine_440)
        assert (zcr >= 0).all()
        assert (zcr <= 1.0).all()


# ---------------------------------------------------------------------------
# onset_strength
# ---------------------------------------------------------------------------

class TestOnsetStrength:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        oenv = feature.onset_strength(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        assert oenv.ndim == 1
        assert len(oenv) > 0

    def test_non_negative(self, sine_440, sr):
        from audiolib import feature
        oenv = feature.onset_strength(y=sine_440, sr=sr)
        assert (oenv >= 0).all()

    def test_from_S(self, stft_matrix, sr):
        from audiolib import feature
        oenv = feature.onset_strength(S=np.abs(stft_matrix), sr=sr)
        assert oenv.ndim == 1
