"""
Tests for audiolib.util — utility functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import HOP_LENGTH, N_FFT, N_SAMPLES


# ---------------------------------------------------------------------------
# valid_audio
# ---------------------------------------------------------------------------

class TestValidAudio:
    def test_valid_mono(self, sine_440):
        from audiolib.util import valid_audio
        assert valid_audio(sine_440) is True

    def test_valid_mono_with_mono_required(self, sine_440):
        from audiolib.util import valid_audio
        assert valid_audio(sine_440, mono=True) is True

    def test_stereo_rejected_when_mono_required(self, stereo_sine):
        from audiolib.util import valid_audio
        with pytest.raises(Exception):
            valid_audio(stereo_sine, mono=True)

    def test_not_ndarray_raises(self):
        from audiolib.util import valid_audio
        with pytest.raises(Exception):
            valid_audio([1.0, 2.0, 3.0])

    def test_integer_dtype_raises(self, sr):
        from audiolib.util import valid_audio
        y = np.zeros(sr, dtype=np.int16)
        with pytest.raises(Exception):
            valid_audio(y)

    def test_zero_dim_raises(self):
        from audiolib.util import valid_audio
        with pytest.raises(Exception):
            valid_audio(np.float32(0.0))

    def test_nan_raises(self, sr):
        from audiolib.util import valid_audio
        y = np.zeros(sr, dtype=np.float32)
        y[100] = np.nan
        with pytest.raises(Exception):
            valid_audio(y)

    def test_inf_raises(self, sr):
        from audiolib.util import valid_audio
        y = np.zeros(sr, dtype=np.float32)
        y[100] = np.inf
        with pytest.raises(Exception):
            valid_audio(y)


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------

class TestFrame:
    def test_basic_shape(self, sine_440):
        from audiolib.util import frame
        f = frame(sine_440, frame_length=2048, hop_length=512)
        n_frames = 1 + (len(sine_440) - 2048) // 512
        assert f.shape == (n_frames, 2048)

    def test_hop_length_1(self, sr):
        from audiolib.util import frame
        x = np.arange(100, dtype=np.float32)
        f = frame(x, frame_length=10, hop_length=1)
        assert f.shape[0] == 91
        assert f.shape[1] == 10

    def test_frame_length_equals_n(self, sr):
        from audiolib.util import frame
        x = np.ones(100, dtype=np.float32)
        f = frame(x, frame_length=100, hop_length=1)
        assert f.shape[0] == 1

    def test_signal_too_short_raises(self):
        from audiolib.util import frame
        x = np.ones(10, dtype=np.float32)
        with pytest.raises(Exception):
            frame(x, frame_length=20, hop_length=5)

    def test_bad_frame_length_raises(self, sine_440):
        from audiolib.util import frame
        with pytest.raises(Exception):
            frame(sine_440, frame_length=0, hop_length=512)

    def test_bad_hop_length_raises(self, sine_440):
        from audiolib.util import frame
        with pytest.raises(Exception):
            frame(sine_440, frame_length=2048, hop_length=0)

    def test_first_frame_content(self, sine_440):
        from audiolib.util import frame
        f = frame(sine_440, frame_length=1024, hop_length=512)
        np.testing.assert_array_equal(f[0], sine_440[:1024])

    def test_last_frame_start(self, sine_440):
        from audiolib.util import frame
        fl, hl = 1024, 512
        f = frame(sine_440, frame_length=fl, hop_length=hl)
        n_frames = f.shape[0]
        last_start = (n_frames - 1) * hl
        np.testing.assert_array_equal(f[-1], sine_440[last_start:last_start + fl])


# ---------------------------------------------------------------------------
# pad_center
# ---------------------------------------------------------------------------

class TestPadCenter:
    def test_pads_to_size(self):
        from audiolib.util import pad_center
        x = np.ones(5, dtype=np.float32)
        out = pad_center(x, size=10)
        assert len(out) == 10

    def test_center_padding(self):
        from audiolib.util import pad_center
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = pad_center(x, size=7)
        assert len(out) == 7
        # Content should be centered
        assert out[2] == 1.0
        assert out[3] == 2.0
        assert out[4] == 3.0

    def test_exact_size_noop(self):
        from audiolib.util import pad_center
        x = np.ones(5, dtype=np.float32)
        out = pad_center(x, size=5)
        np.testing.assert_array_equal(out, x)

    def test_smaller_size_raises(self):
        from audiolib.util import pad_center
        x = np.ones(10, dtype=np.float32)
        with pytest.raises(Exception):
            pad_center(x, size=5)

    def test_2d_array_axis(self):
        from audiolib.util import pad_center
        x = np.ones((3, 5), dtype=np.float32)
        out = pad_center(x, size=10, axis=-1)
        assert out.shape == (3, 10)


# ---------------------------------------------------------------------------
# fix_length
# ---------------------------------------------------------------------------

class TestFixLength:
    def test_truncate(self):
        from audiolib.util import fix_length
        x = np.ones(20, dtype=np.float32)
        out = fix_length(x, size=10)
        assert len(out) == 10
        np.testing.assert_array_equal(out, x[:10])

    def test_extend(self):
        from audiolib.util import fix_length
        x = np.ones(5, dtype=np.float32)
        out = fix_length(x, size=10)
        assert len(out) == 10
        np.testing.assert_array_equal(out[:5], x)
        np.testing.assert_array_equal(out[5:], np.zeros(5))

    def test_exact_size_noop(self):
        from audiolib.util import fix_length
        x = np.arange(10, dtype=np.float32)
        out = fix_length(x, size=10)
        np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_inf_norm(self):
        from audiolib.util import normalize
        x = np.array([[1.0, 2.0, 4.0], [0.5, 1.0, 2.0]], dtype=np.float32)
        out = normalize(x, norm=np.inf)
        assert np.max(np.abs(out), axis=0).max() <= 1.0 + 1e-6

    def test_l1_norm(self):
        from audiolib.util import normalize
        x = np.abs(np.random.randn(4, 10).astype(np.float32))
        out = normalize(x, norm=1)
        col_sums = np.sum(np.abs(out), axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-5)

    def test_l2_norm(self):
        from audiolib.util import normalize
        x = np.random.randn(4, 10).astype(np.float32)
        out = normalize(x, norm=2)
        col_l2 = np.sqrt(np.sum(out ** 2, axis=0))
        np.testing.assert_allclose(col_l2, 1.0, atol=1e-5)

    def test_none_norm_noop(self):
        from audiolib.util import normalize
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = normalize(x, norm=None)
        np.testing.assert_array_equal(out, x)

    def test_zero_column_handled(self):
        from audiolib.util import normalize
        x = np.zeros((4, 5), dtype=np.float32)
        out = normalize(x, norm=np.inf)
        # Should not raise, zero columns get 1.0 divisor
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# tiny
# ---------------------------------------------------------------------------

class TestTiny:
    def test_float32_tiny(self):
        from audiolib.util import tiny
        t = tiny(np.float32(0.0))
        assert 0.0 < t < 1e-30

    def test_float64_tiny(self):
        from audiolib.util import tiny
        t = tiny(np.float64(0.0))
        assert 0.0 < t < 1e-300

    def test_float32_array(self):
        from audiolib.util import tiny
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = tiny(arr)
        assert t == np.finfo(np.float32).tiny


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------

class TestStack:
    def test_stack_axis0(self):
        from audiolib.util import stack
        a = np.ones((3, 4), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32) * 2
        out = stack([a, b], axis=0)
        assert out.shape == (2, 3, 4)

    def test_stack_axis1(self):
        from audiolib.util import stack
        a = np.ones((3, 4), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32) * 2
        out = stack([a, b], axis=1)
        assert out.shape == (3, 2, 4)


# ---------------------------------------------------------------------------
# axis_sort
# ---------------------------------------------------------------------------

class TestAxisSort:
    def test_sorts_columns(self):
        from audiolib.util import axis_sort
        x = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float32)
        out = axis_sort(x)
        # Should return valid array
        assert out.shape == x.shape

    def test_returns_index(self):
        from audiolib.util import axis_sort
        x = np.random.rand(4, 6).astype(np.float32)
        out, idx = axis_sort(x, index=True)
        assert idx.shape[0] == x.shape[1]


# ---------------------------------------------------------------------------
# softmask
# ---------------------------------------------------------------------------

class TestSoftmask:
    def test_values_in_0_1(self):
        from audiolib.util import softmask
        X = np.random.rand(5, 10).astype(np.float32)
        X_ref = np.random.rand(5, 10).astype(np.float32)
        mask = softmask(X, X_ref)
        assert mask.min() >= 0.0 - 1e-6
        assert mask.max() <= 1.0 + 1e-6

    def test_shape_preserved(self):
        from audiolib.util import softmask
        X = np.ones((4, 6), dtype=np.float32)
        mask = softmask(X, X)
        assert mask.shape == X.shape

    def test_shape_mismatch_raises(self):
        from audiolib.util import softmask
        X = np.ones((4, 6), dtype=np.float32)
        X_ref = np.ones((4, 5), dtype=np.float32)
        with pytest.raises(Exception):
            softmask(X, X_ref)

    def test_power_inf(self):
        from audiolib.util import softmask
        X = np.array([[0.8, 0.3]], dtype=np.float32)
        X_ref = np.array([[0.5, 0.5]], dtype=np.float32)
        mask = softmask(X, X_ref, power=np.inf)
        # X > X_ref → mask=1; X < X_ref → mask=0
        assert mask[0, 0] == 1.0
        assert mask[0, 1] == 0.0

    def test_bad_power_raises(self):
        from audiolib.util import softmask
        X = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(Exception):
            softmask(X, X, power=0.0)


# ---------------------------------------------------------------------------
# localmax
# ---------------------------------------------------------------------------

class TestLocalmax:
    def test_basic_local_max(self):
        from audiolib.util import localmax
        x = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 0.0], dtype=np.float32)
        lm = localmax(x)
        # x[1]=1.0 is local max; x[3]=2.0 is local max
        assert lm[1]
        assert lm[3]
        assert not lm[0]
        assert not lm[5]

    def test_monotone_no_max(self):
        from audiolib.util import localmax
        x = np.arange(10, dtype=np.float32)
        lm = localmax(x)
        # Only the last element could be a max (boundary)
        assert lm.sum() <= 1

    def test_output_dtype_bool(self):
        from audiolib.util import localmax
        x = np.random.rand(20).astype(np.float32)
        lm = localmax(x)
        assert lm.dtype == bool

    def test_shape_preserved(self):
        from audiolib.util import localmax
        x = np.random.rand(4, 10).astype(np.float32)
        lm = localmax(x)
        assert lm.shape == x.shape


# ---------------------------------------------------------------------------
# sparsify_rows
# ---------------------------------------------------------------------------

class TestSparsifyRows:
    def test_output_type(self):
        pytest.importorskip("scipy")
        from audiolib.util import sparsify_rows
        x = np.random.rand(4, 10).astype(np.float32)
        S = sparsify_rows(x)
        import scipy.sparse
        assert scipy.sparse.issparse(S)

    def test_output_shape(self):
        pytest.importorskip("scipy")
        from audiolib.util import sparsify_rows
        x = np.random.rand(5, 8).astype(np.float32)
        S = sparsify_rows(x)
        assert S.shape == x.shape

    def test_zeros_out_small_values(self):
        pytest.importorskip("scipy")
        from audiolib.util import sparsify_rows
        x = np.ones((3, 10), dtype=np.float32)
        x[0, 0] = 0.0001
        S = sparsify_rows(x, quantile=0.5)
        # After sparsification, at least some zeros
        assert S.nnz < x.size
