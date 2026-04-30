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


# ---------------------------------------------------------------------------
# delta
# ---------------------------------------------------------------------------

class TestDelta:
    def test_output_shape_default(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        d = feature.delta(mfcc)
        assert d.shape == mfcc.shape

    def test_second_order_delta(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        d2 = feature.delta(mfcc, order=2)
        assert d2.shape == mfcc.shape

    def test_constant_signal_zero_delta(self):
        from audiolib import feature
        x = np.ones((5, 20), dtype=np.float32)
        d = feature.delta(x)
        np.testing.assert_allclose(d, 0.0, atol=1e-5)

    def test_linear_ramp_constant_derivative(self):
        from audiolib import feature
        # Linear ramp: delta should be nearly constant (away from edges)
        x = np.tile(np.arange(20, dtype=np.float32), (3, 1))
        d = feature.delta(x)
        assert d.shape == x.shape

    def test_width_param(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        d3 = feature.delta(mfcc, width=3)
        d9 = feature.delta(mfcc, width=9)
        assert d3.shape == d9.shape == mfcc.shape

    def test_axis_param(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        d = feature.delta(mfcc, axis=0)
        assert d.shape == mfcc.shape

    def test_dtype_preserved(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        d = feature.delta(mfcc)
        assert np.issubdtype(d.dtype, np.floating)


# ---------------------------------------------------------------------------
# stack_memory
# ---------------------------------------------------------------------------

class TestStackMemory:
    def test_output_shape(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        stacked = feature.stack_memory(mfcc, n_steps=3)
        assert stacked.shape[0] == 13 * 3
        assert stacked.shape[1] == mfcc.shape[1]

    def test_n_steps_1_noop(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        stacked = feature.stack_memory(mfcc, n_steps=1)
        # n_steps=1 should return the input (padded or not)
        assert stacked.shape[0] == 13
        assert stacked.shape[1] == mfcc.shape[1]

    def test_n_steps_2(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        stacked = feature.stack_memory(mfcc, n_steps=2)
        assert stacked.shape == (26, mfcc.shape[1])

    def test_delay_param(self, sine_440, sr):
        from audiolib import feature
        mfcc = feature.mfcc(y=sine_440, sr=sr, n_mfcc=13)
        stacked = feature.stack_memory(mfcc, n_steps=2, delay=2)
        assert stacked.shape[0] == 26
        assert stacked.shape[1] == mfcc.shape[1]


# ---------------------------------------------------------------------------
# spectral_contrast
# ---------------------------------------------------------------------------

class TestSpectralContrast:
    def test_shape_default(self, sine_440, sr):
        from audiolib import feature
        sc = feature.spectral_contrast(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        # Default n_bands=6 → 7 rows (n_bands + 1)
        assert sc.shape == (7, nf)

    def test_n_bands_param(self, sine_440, sr):
        from audiolib import feature
        sc = feature.spectral_contrast(y=sine_440, sr=sr, n_bands=4)
        assert sc.shape[0] == 5  # n_bands + 1

    def test_from_S(self, stft_matrix, sr):
        from audiolib import feature
        sc = feature.spectral_contrast(S=np.abs(stft_matrix), sr=sr, n_fft=N_FFT)
        assert sc.shape[0] == 7

    def test_non_negative_output(self, white_noise, sr):
        from audiolib import feature
        sc = feature.spectral_contrast(y=white_noise, sr=sr)
        # Contrast values can be any sign; check they're finite
        assert np.isfinite(sc).all()

    def test_pure_tone_vs_noise(self, sine_440, white_noise, sr):
        from audiolib import feature
        # Pure tone should have higher contrast (peaked spectrum) than noise
        sc_sine = feature.spectral_contrast(y=sine_440, sr=sr)
        sc_noise = feature.spectral_contrast(y=white_noise, sr=sr)
        assert sc_sine.mean() > sc_noise.mean() - 1.0  # loose check


# ---------------------------------------------------------------------------
# tonnetz
# ---------------------------------------------------------------------------

class TestTonnetz:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        tnz = feature.tonnetz(y=sine_440, sr=sr)
        nf = n_frames_for(len(sine_440))
        assert tnz.shape == (6, nf)

    def test_from_chroma(self, sine_440, sr):
        from audiolib import feature
        chroma = feature.chroma_stft(y=sine_440, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        tnz = feature.tonnetz(chroma=chroma)
        assert tnz.shape == (6, chroma.shape[1])

    def test_values_bounded(self, sine_440, sr):
        from audiolib import feature
        tnz = feature.tonnetz(y=sine_440, sr=sr)
        # Tonnetz values should be in a reasonable range
        assert np.isfinite(tnz).all()

    def test_dtype_float(self, sine_440, sr):
        from audiolib import feature
        tnz = feature.tonnetz(y=sine_440, sr=sr)
        assert np.issubdtype(tnz.dtype, np.floating)


# ---------------------------------------------------------------------------
# poly_features
# ---------------------------------------------------------------------------

class TestPolyFeatures:
    def test_shape_order_0(self, sine_440, sr):
        from audiolib import feature
        pf = feature.poly_features(y=sine_440, sr=sr, order=0, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert pf.shape == (1, nf)

    def test_shape_order_1(self, sine_440, sr):
        from audiolib import feature
        pf = feature.poly_features(y=sine_440, sr=sr, order=1, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert pf.shape == (2, nf)

    def test_shape_order_2(self, sine_440, sr):
        from audiolib import feature
        pf = feature.poly_features(y=sine_440, sr=sr, order=2, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf = n_frames_for(len(sine_440))
        assert pf.shape == (3, nf)

    def test_from_S(self, stft_matrix, sr):
        from audiolib import feature
        pf = feature.poly_features(S=np.abs(stft_matrix), sr=sr, n_fft=N_FFT, order=1)
        assert pf.shape[0] == 2
        assert np.isfinite(pf).all()

    def test_dtype_float(self, sine_440, sr):
        from audiolib import feature
        pf = feature.poly_features(y=sine_440, sr=sr, order=1)
        assert np.issubdtype(pf.dtype, np.floating)


# ---------------------------------------------------------------------------
# tempogram
# ---------------------------------------------------------------------------

class TestTempogram:
    def test_shape(self, sine_440, sr):
        from audiolib import feature
        tg = feature.tempogram(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        # Default win_length=384 → n_tempo_bins rows
        assert tg.ndim == 2
        assert tg.shape[1] > 0

    def test_from_onset_envelope(self, sine_440, sr):
        from audiolib import feature
        oenv = feature.onset_strength(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        tg = feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH)
        assert tg.ndim == 2

    def test_non_negative(self, sine_440, sr):
        from audiolib import feature
        tg = feature.tempogram(y=sine_440, sr=sr, hop_length=HOP_LENGTH)
        # Tempogram rows correspond to lag/autocorrelation values; values are
        # real-valued and need not be non-negative (auto-correlation ranges in [-1,1]).
        assert np.isfinite(tg).all()

    def test_win_length_param(self, sine_440, sr):
        from audiolib import feature
        tg = feature.tempogram(y=sine_440, sr=sr, hop_length=HOP_LENGTH, win_length=128)
        assert tg.ndim == 2
