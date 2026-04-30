# Changelog

All notable changes to **audiolib** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2024-01-01

### Added

- Initial release of **audiolib** — a Rust-backed, drop-in replacement for [librosa](https://librosa.org/).
- **Core audio I/O** (`audiolib.core`):
  - `load` — load audio files via soundfile, with optional resampling and mono conversion.
  - `get_duration`, `get_samplerate` — metadata helpers.
  - `to_mono`, `resample` — powered by Rust for maximum throughput.
  - `stft`, `istft` — short-time Fourier transform and inverse via rustfft.
  - `magphase`, `amplitude_to_db`, `power_to_db`, `db_to_amplitude`, `db_to_power`.
  - `zero_crossings`, `autocorrelate` — Rust-accelerated.
  - `mu_compress`, `mu_expand` — μ-law companding.
  - `clicks`, `tone`, `chirp` — signal generators.
- **Feature extraction** (`audiolib.feature`):
  - `melspectrogram`, `mfcc` — with both Slaney and HTK mel scales.
  - `chroma_stft` — 12-bin L2-normalized chroma features.
  - `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`.
  - `rms`, `zero_crossing_rate`, `onset_strength`.
- **Unit conversions** (`audiolib.convert`):
  - `hz_to_mel`, `mel_to_hz` — with HTK mode.
  - `hz_to_midi`, `midi_to_hz`, `note_to_midi`, `midi_to_note`, `note_to_hz`, `hz_to_note`.
  - `frames_to_samples`, `samples_to_frames`, `frames_to_time`, `time_to_frames`,
    `samples_to_time`, `time_to_samples`.
  - `fft_frequencies`, `mel_frequencies`.
- **Effects** (`audiolib.effects`):
  - `time_stretch`, `pitch_shift` — phase-vocoder based.
  - `trim`, `split` — silence detection.
  - `harmonic`, `percussive`, `hpss` — harmonic/percussive source separation.
  - `remix`, `preemphasis`, `deemphasis`.
- **Utilities** (`audiolib.util`):
  - `valid_audio`, `frame`, `pad_center`, `fix_length`, `normalize`, `tiny`,
    `stack`, `axis_sort`, `softmask`, `localmax`, `sparsify_rows`.
- Full test suite (`tests/`) with coverage for all modules.
- GitHub Actions CI: rust-check, python-lint, python-tests (Python 3.8–3.13 × 3 OSes).
- GitHub Actions publish workflow with OIDC trusted publishing to PyPI/TestPyPI.
- Performance benchmark suite (`tests/benchmark.py`).
