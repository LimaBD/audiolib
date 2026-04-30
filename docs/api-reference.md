# API Reference

All public functions in audiolib, organized by module. Parameter types follow NumPy conventions — `np.ndarray` arrays are always float32 unless noted.

---

## `audiolib` / `audiolib.core`

### `load`

```python
audiolib.load(
    path: str,
    *,
    sr: Optional[float] = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
    dtype = np.float32,
    res_type: str = "soxr_hq",
) -> Tuple[np.ndarray, int]
```

Load an audio file from disk.

| Parameter | Description |
|---|---|
| `path` | Path to audio file (any format supported by soundfile) |
| `sr` | Target sample rate. `None` = keep native. Default `22050` |
| `mono` | Convert to mono. Default `True` |
| `offset` | Start position in seconds |
| `duration` | Duration to load in seconds. `None` = entire file |
| `dtype` | Output dtype. Default `np.float32` |
| `res_type` | Resampling algorithm (`"soxr_hq"`, `"soxr_mq"`, `"soxr_lq"`, `"linear"`) |

Returns `(y, sr)` where `y` is a 1-D float32 array and `sr` is the effective sample rate.

---

### `stft`

```python
audiolib.stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,      # default: n_fft // 4
    win_length: Optional[int] = None,      # default: n_fft
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray  # shape: (1 + n_fft/2, n_frames), complex64
```

Short-Time Fourier Transform computed by the Rust engine using rustfft.

---

### `istft`

```python
audiolib.istft(
    stft_matrix: np.ndarray,
    *,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    length: Optional[int] = None,
) -> np.ndarray  # shape: (n_samples,), float32
```

Inverse STFT with overlap-add reconstruction.

---

### `magphase`

```python
audiolib.magphase(D: np.ndarray, *, power: float = 1) -> Tuple[np.ndarray, np.ndarray]
```

Decompose a complex STFT matrix into magnitude and unit-circle phase.

---

### `amplitude_to_db` / `db_to_amplitude`

```python
audiolib.amplitude_to_db(S: np.ndarray, *, ref=1.0, amin: float = 1e-5, top_db: Optional[float] = 80.0) -> np.ndarray
audiolib.db_to_amplitude(S_db: np.ndarray, *, ref: float = 1.0) -> np.ndarray
```

### `power_to_db` / `db_to_power`

```python
audiolib.power_to_db(S: np.ndarray, *, ref=1.0, amin: float = 1e-10, top_db: Optional[float] = 80.0) -> np.ndarray
audiolib.db_to_power(S_db: np.ndarray, *, ref: float = 1.0) -> np.ndarray
```

Convert between linear magnitude/power and decibels. All operations run in Rust.

---

### `to_mono`

```python
audiolib.to_mono(y: np.ndarray) -> np.ndarray
```

Average stereo or multi-channel audio down to mono. Input shape: `(channels, samples)` or `(samples,)`.

---

### `resample`

```python
audiolib.resample(
    y: np.ndarray,
    *,
    orig_sr: float,
    target_sr: float,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
) -> np.ndarray
```

Resample a signal from `orig_sr` to `target_sr`. Uses soxr when available, falls back to the Rust linear interpolation resampler.

---

### `get_duration`

```python
audiolib.get_duration(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    path: Optional[str] = None,
) -> float
```

Get duration in seconds from a signal, a spectrogram, or a file path.

---

### `get_samplerate`

```python
audiolib.get_samplerate(path: str) -> int
```

Read the native sample rate of an audio file without loading its content.

---

### `zero_crossings`

```python
audiolib.zero_crossings(
    y: np.ndarray,
    *,
    threshold: float = 1e-10,
    pad: bool = True,
) -> np.ndarray  # bool, same shape as y
```

Return a boolean array indicating where the signal crosses zero.

---

### `autocorrelate`

```python
audiolib.autocorrelate(y: np.ndarray, *, max_size: Optional[int] = None) -> np.ndarray
```

Compute the normalized autocorrelation of a 1-D signal using FFT-based convolution in Rust.

---

### `mu_compress` / `mu_expand`

```python
audiolib.mu_compress(x: np.ndarray, *, mu: int = 255, quantize: bool = False) -> np.ndarray
audiolib.mu_expand(x: np.ndarray, *, mu: int = 255, quantize: bool = False) -> np.ndarray
```

μ-law companding (compressing and expanding). Useful for audio quantization.

---

### Signal generators

```python
audiolib.tone(frequency: float, *, sr: int = 22050, length: Optional[int] = None, duration: float = 1.0, phi: float = 0.0) -> np.ndarray
audiolib.chirp(fmin: float, fmax: float, *, sr: int = 22050, length: Optional[int] = None, duration: float = 1.0, linear: bool = False, phi: float = 0.0) -> np.ndarray
audiolib.clicks(*, times=None, frames=None, sr: int = 22050, hop_length: int = 512, click_freq: float = 1000.0, click_duration: float = 0.1, click=None, length: Optional[int] = None) -> np.ndarray
```

---

## `audiolib.feature`

All functions accept either raw audio `y` + `sr`, or a pre-computed power/magnitude spectrogram `S`.

### `melspectrogram`

```python
audiolib.feature.melspectrogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    power: float = 2.0,
) -> np.ndarray  # shape: (n_mels, n_frames)
```

Mel-frequency power spectrogram. The mel filter bank is computed and applied in Rust.

---

### `mfcc`

```python
audiolib.feature.mfcc(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: str = "ortho",
    lifter: float = 0,
    **kwargs,
) -> np.ndarray  # shape: (n_mfcc, n_frames)
```

Mel-frequency cepstral coefficients. DCT-II is computed in Rust.

---

### `chroma_stft`

```python
audiolib.feature.chroma_stft(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    tuning: float = 0.0,
    n_chroma: int = 12,
    norm: float = 2,
    **kwargs,
) -> np.ndarray  # shape: (12, n_frames)
```

12-bin L2-normalized pitch class energy profile (chromagram).

---

### `spectral_centroid`

```python
audiolib.feature.spectral_centroid(
    *, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, **kwargs
) -> np.ndarray  # shape: (1, n_frames)
```

Frequency-weighted center of mass of the spectrum — a measure of spectral brightness.

---

### `spectral_bandwidth`

```python
audiolib.feature.spectral_bandwidth(
    *, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, centroid=None, **kwargs
) -> np.ndarray  # shape: (1, n_frames)
```

---

### `spectral_rolloff`

```python
audiolib.feature.spectral_rolloff(
    *, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, roll_percent=0.85, freq=None, **kwargs
) -> np.ndarray  # shape: (1, n_frames)
```

Frequency below which `roll_percent` (default 85%) of the spectral energy is concentrated.

---

### `spectral_flatness`

```python
audiolib.feature.spectral_flatness(
    *, y=None, S=None, n_fft=2048, hop_length=512, amin=1e-10, power=2.0, **kwargs
) -> np.ndarray  # shape: (1, n_frames)
```

Ratio of geometric mean to arithmetic mean of the spectrum. Returns 1 for white noise, 0 for a pure tone.

---

### `rms`

```python
audiolib.feature.rms(
    *, y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode="reflect"
) -> np.ndarray  # shape: (1, n_frames)
```

Root mean square energy per frame.

---

### `zero_crossing_rate`

```python
audiolib.feature.zero_crossing_rate(
    y: np.ndarray, *, frame_length=2048, hop_length=512, center=True
) -> np.ndarray  # shape: (1, n_frames)
```

---

### `onset_strength`

```python
audiolib.feature.onset_strength(
    *, y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
    aggregate=None, detrend=False, centering=True, **kwargs,
) -> np.ndarray  # shape: (n_frames,)
```

Spectral flux onset strength envelope (half-wave rectified, computed in Rust).

---

## `audiolib.convert`

All conversion functions accept scalars, lists, or NumPy arrays.

### Frequency conversions

```python
audiolib.hz_to_mel(frequencies, *, htk: bool = False)
audiolib.mel_to_hz(mels, *, htk: bool = False)
audiolib.hz_to_midi(frequencies)
audiolib.midi_to_hz(midi)
audiolib.note_to_midi(note: str) -> int          # "A4" → 69
audiolib.midi_to_note(midi: int) -> str          # 69 → "A4"
audiolib.note_to_hz(note: str) -> float          # "A4" → 440.0
audiolib.hz_to_note(frequencies) -> str | list   # 440.0 → "A4"
```

### Frame / sample / time

```python
audiolib.frames_to_samples(frames, *, hop_length: int = 512, n_fft: int = 0)
audiolib.samples_to_frames(samples, *, hop_length: int = 512, n_fft: int = 0)
audiolib.frames_to_time(frames, *, sr: int = 22050, hop_length: int = 512, n_fft: int = 0)
audiolib.time_to_frames(times, *, sr: int = 22050, hop_length: int = 512, n_fft: int = 0)
audiolib.samples_to_time(samples, *, sr: int = 22050)
audiolib.time_to_samples(times, *, sr: int = 22050)
```

### Frequency grids

```python
audiolib.fft_frequencies(*, sr: int = 22050, n_fft: int = 2048) -> np.ndarray  # (1 + n_fft/2,)
audiolib.mel_frequencies(*, n_mels: int = 128, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False) -> np.ndarray
```

---

## `audiolib.effects`

### `time_stretch`

```python
audiolib.effects.time_stretch(y: np.ndarray, *, rate: float, **kwargs) -> np.ndarray
```

Phase-vocoder time stretching. `rate > 1` speeds up, `rate < 1` slows down.

---

### `pitch_shift`

```python
audiolib.effects.pitch_shift(y: np.ndarray, *, sr: float, n_steps: float, bins_per_octave: int = 12, res_type: str = "soxr_hq", **kwargs) -> np.ndarray
```

Shift pitch by `n_steps` semitones (positive = up, negative = down).

---

### `trim` / `split`

```python
audiolib.effects.trim(y, *, top_db=60, frame_length=2048, hop_length=512) -> Tuple[np.ndarray, np.ndarray]
audiolib.effects.split(y, *, top_db=60, frame_length=2048, hop_length=512) -> np.ndarray  # shape: (n_intervals, 2)
```

Remove or identify silent regions. `trim` returns the trimmed signal and `[start, end]` sample indices.

---

### `hpss` / `harmonic` / `percussive`

```python
audiolib.effects.hpss(y, *, kernel_size=31, power=2.0, mask=False, margin=1.0, n_fft=2048, hop_length=512) -> Tuple[np.ndarray, np.ndarray]
audiolib.effects.harmonic(y, **kwargs) -> np.ndarray
audiolib.effects.percussive(y, **kwargs) -> np.ndarray
```

Harmonic-percussive source separation via median filtering on the spectrogram. Requires scipy.

---

### `preemphasis` / `deemphasis`

```python
audiolib.effects.preemphasis(y, *, coef=0.97, return_zf=False)
audiolib.effects.deemphasis(y, *, coef=0.97, return_zf=False)
```

First-order IIR pre/de-emphasis filter. `return_zf=True` returns state for streaming use.

---

## `audiolib.util`

```python
audiolib.util.valid_audio(y, *, mono=False) -> bool           # raises ParameterError if invalid
audiolib.util.frame(x, *, frame_length, hop_length, axis=-1)  # strided view, no copy
audiolib.util.pad_center(data, *, size, axis=-1)              # center-pad to target size
audiolib.util.fix_length(data, *, size, axis=-1)              # trim or zero-pad
audiolib.util.normalize(S, *, norm=np.inf, axis=0, threshold=None, fill=None)
audiolib.util.tiny(x) -> float                                 # smallest positive value for dtype
audiolib.util.stack(arrays, *, axis=0) -> np.ndarray
audiolib.util.axis_sort(S, *, axis=-1, index=False, value=None)
audiolib.util.softmask(X, X_ref, *, power=1.0, split_zeros=False) -> np.ndarray
audiolib.util.localmax(x, *, axis=0) -> np.ndarray             # bool mask of local maxima
audiolib.util.sparsify_rows(x, *, quantile=0.01)               # scipy sparse CSR matrix
```

---

## Exceptions

```python
from audiolib.exceptions import AudiolibError, ParameterError
```

`ParameterError` is raised for invalid parameter values (bad shape, out-of-range values, etc.). `AudiolibError` is the base class for all library exceptions.
