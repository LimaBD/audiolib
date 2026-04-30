"""
audiolib — blazing-fast audio analysis and DSP for Python.

Process audio at native speed with a Rust engine under the hood.
STFT, mel spectrograms, MFCCs, chroma, spectral features, effects —
all compiled to machine code via PyO3, all accessible from Python:

    import audiolib

    y, sr = audiolib.load("audio.wav")
    S = audiolib.stft(y)
    mel = audiolib.feature.melspectrogram(y=y, sr=sr)
    mfccs = audiolib.feature.mfcc(y=y, sr=sr)
"""

from audiolib.core import (
    load,
    get_duration,
    get_samplerate,
    to_mono,
    resample,
    stft,
    istft,
    magphase,
    amplitude_to_db,
    db_to_amplitude,
    power_to_db,
    db_to_power,
    zero_crossings,
    autocorrelate,
    mu_compress,
    mu_expand,
    clicks,
    tone,
    chirp,
)
from audiolib.convert import (
    hz_to_mel,
    mel_to_hz,
    hz_to_midi,
    midi_to_hz,
    midi_to_note,
    note_to_hz,
    note_to_midi,
    hz_to_note,
    frames_to_samples,
    frames_to_time,
    samples_to_frames,
    samples_to_time,
    time_to_frames,
    time_to_samples,
    fft_frequencies,
    mel_frequencies,
)
from audiolib import feature, effects, util
from audiolib.exceptions import AudiolibError, ParameterError

__version__ = "0.1.0"
__author__ = "Bruno Lima"
__license__ = "MIT"

__all__ = [
    # Core audio I/O
    "load",
    "get_duration",
    "get_samplerate",
    "to_mono",
    "resample",
    # Spectral
    "stft",
    "istft",
    "magphase",
    # Magnitude scaling
    "amplitude_to_db",
    "db_to_amplitude",
    "power_to_db",
    "db_to_power",
    # Time-domain
    "zero_crossings",
    "autocorrelate",
    "mu_compress",
    "mu_expand",
    # Signal generation
    "clicks",
    "tone",
    "chirp",
    # Frequency/unit conversions
    "hz_to_mel",
    "mel_to_hz",
    "hz_to_midi",
    "midi_to_hz",
    "midi_to_note",
    "note_to_hz",
    "note_to_midi",
    "hz_to_note",
    "frames_to_samples",
    "frames_to_time",
    "samples_to_frames",
    "samples_to_time",
    "time_to_frames",
    "time_to_samples",
    "fft_frequencies",
    "mel_frequencies",
    # Sub-modules
    "feature",
    "effects",
    "util",
    # Exceptions
    "AudiolibError",
    "ParameterError",
    # Version
    "__version__",
]
