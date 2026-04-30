"""
audiolib.util — Utility functions for audio processing.

API-compatible with librosa.util.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from audiolib.exceptions import ParameterError

__all__ = [
    "valid_audio",
    "frame",
    "pad_center",
    "fix_length",
    "normalize",
    "tiny",
    "stack",
    "axis_sort",
    "softmask",
    "localmax",
    "sparsify_rows",
]


def valid_audio(y: np.ndarray, *, mono: bool = False) -> bool:
    """Validate whether a variable contains a valid, non-empty audio signal.

    API-compatible with ``librosa.util.valid_audio``.

    Parameters
    ----------
    y : np.ndarray
        Audio data.
    mono : bool
        Whether to require mono (1-D) audio.

    Returns
    -------
    valid : bool

    Raises
    ------
    ParameterError
        If y is not a valid audio array.
    """
    if not isinstance(y, np.ndarray):
        raise ParameterError(f"Audio data must be of type np.ndarray, got {type(y)}")
    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError(f"Audio must be floating-point, got {y.dtype}")
    if y.ndim == 0:
        raise ParameterError("Audio must be at least 1-D")
    if mono and y.ndim != 1:
        raise ParameterError(f"Audio must be mono (1-D), got shape {y.shape}")
    if not np.isfinite(y).all():
        raise ParameterError("Audio contains non-finite values (NaN or Inf)")
    return True


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    """Slice a signal into overlapping frames.

    API-compatible with ``librosa.util.frame``.

    Parameters
    ----------
    x : np.ndarray
    frame_length : int
    hop_length : int
    axis : int
        The axis along which to frame.

    Returns
    -------
    x_frames : np.ndarray
        Framed view of x. Shape: ``x.shape[:axis] + (frame_length, n_frames)``.
    """
    if frame_length <= 0:
        raise ParameterError("frame_length must be > 0")
    if hop_length <= 0:
        raise ParameterError("hop_length must be > 0")

    n = x.shape[axis]
    if n < frame_length:
        raise ParameterError(
            f"Input length ({n}) must be >= frame_length ({frame_length})"
        )

    n_frames = 1 + (n - frame_length) // hop_length
    strides = list(x.strides)
    shape = list(x.shape)

    out_strides = strides[:axis] + [strides[axis] * hop_length, strides[axis]] + strides[axis + 1:]
    out_shape = shape[:axis] + [n_frames, frame_length] + shape[axis + 1:]

    out = np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides, subok=subok, writeable=writeable)
    # Move the frames axis to the last position then frame_length before it
    # Result shape: (..., frame_length, n_frames) for axis=-1
    if axis != -1 and axis != x.ndim - 1:
        out = np.moveaxis(out, axis, -2)
    return out


def pad_center(
    data: np.ndarray,
    *,
    size: int,
    axis: int = -1,
    **kwargs,
) -> np.ndarray:
    """Pad an array to a target size, centering the data.

    API-compatible with ``librosa.util.pad_center``.
    """
    n = data.shape[axis]
    if n > size:
        raise ParameterError(
            f"Target size ({size}) must be >= input size ({n})"
        )
    lpad = (size - n) // 2
    rpad = size - n - lpad
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, rpad)
    return np.pad(data, lengths, **kwargs)


def fix_length(
    data: np.ndarray,
    *,
    size: int,
    axis: int = -1,
    **kwargs,
) -> np.ndarray:
    """Fix the length of a 1-D array by either trimming or zero-padding.

    API-compatible with ``librosa.util.fix_length``.
    """
    n = data.shape[axis]
    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]
    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)
    return data


def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    """Normalize an array along a chosen axis.

    API-compatible with ``librosa.util.normalize``.
    """
    if threshold is None:
        threshold = tiny(S)

    if norm is None:
        return S

    if norm == np.inf:
        length = np.max(np.abs(S), axis=axis, keepdims=True)
    elif norm == -np.inf:
        length = np.min(np.abs(S), axis=axis, keepdims=True)
    elif norm == 0:
        length = np.sum(S != 0, axis=axis, keepdims=True).astype(S.dtype)
    else:
        length = np.sum(np.abs(S) ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

    small_idx = length < threshold

    if fill is None:
        length[small_idx] = 1.0
    elif fill:
        length[small_idx] = threshold
    else:
        length[small_idx] = 1.0

    return S / length


def tiny(x: np.ndarray) -> float:
    """The smallest positive floating-point number for x's dtype.

    API-compatible with ``librosa.util.tiny``.
    """
    if np.issubdtype(type(x), np.floating):
        dtype = type(x)
    elif isinstance(x, np.ndarray):
        dtype = x.dtype
    else:
        dtype = np.float32
    return np.finfo(dtype).tiny


def stack(
    arrays,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Stack a sequence of arrays along a new axis.

    Thin wrapper around ``np.stack`` with librosa-compatible name.
    API-compatible with ``librosa.util.stack``.
    """
    return np.stack(arrays, axis=axis)


def axis_sort(
    S: np.ndarray,
    *,
    axis: int = -1,
    index: bool = False,
    value=None,
) -> np.ndarray | tuple:
    """Sort an array along a given axis.

    API-compatible with ``librosa.util.axis_sort``.
    """
    if value is None:
        value = np.argmax

    bin_idx = value(S, axis=axis)
    idx = np.argsort(bin_idx)

    S_sorted = S[:, idx] if axis == -1 or axis == S.ndim - 1 else S[idx]

    if index:
        return S_sorted, idx
    return S_sorted


def softmask(
    X: np.ndarray,
    X_ref: np.ndarray,
    *,
    power: float = 1.0,
    split_zeros: bool = False,
) -> np.ndarray:
    """Compute a soft-mask for filtering.

    API-compatible with ``librosa.util.softmask``.
    """
    if X.shape != X_ref.shape:
        raise ParameterError("X and X_ref must have the same shape")
    if power <= 0:
        raise ParameterError("power must be positive")

    if np.isinf(power):
        mask = X_ref <= X
    else:
        Xp = X ** power
        Rp = X_ref ** power
        denom = Xp + Rp
        if split_zeros:
            mask = np.where(denom < tiny(X), 0.5, Xp / denom)
        else:
            mask = np.where(denom < tiny(X), 0.0, Xp / denom)

    return mask.astype(np.float32)


def localmax(
    x: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Find local maxima in an array.

    API-compatible with ``librosa.util.localmax``.
    """
    padded = np.pad(x, [(1, 1) if i == axis else (0, 0) for i in range(x.ndim)], mode="edge")
    slices_prev = [slice(None)] * x.ndim
    slices_next = [slice(None)] * x.ndim
    slices_prev[axis] = slice(0, x.shape[axis])
    slices_next[axis] = slice(2, x.shape[axis] + 2)
    return (x > padded[tuple(slices_prev)]) & (x >= padded[tuple(slices_next)])


def sparsify_rows(
    x: np.ndarray,
    *,
    quantile: float = 0.01,
    dtype=None,
) -> scipy.sparse.csr_matrix:  # type: ignore[name-defined]  # noqa: F821
    """Return a sparse row-matrix where values below a threshold are zeroed.

    API-compatible with ``librosa.util.sparsify_rows``.
    """
    try:
        import scipy.sparse
    except ImportError as exc:
        raise ImportError("scipy is required for sparsify_rows") from exc

    if dtype is None:
        dtype = x.dtype

    x_sparse = x.copy().astype(dtype)
    for i, row in enumerate(x_sparse):
        threshold = np.percentile(np.abs(row), 100 * (1 - quantile))
        x_sparse[i, np.abs(row) < threshold] = 0

    return scipy.sparse.csr_matrix(x_sparse)
