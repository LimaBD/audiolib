"""
audiolib.convert — Unit and frequency conversion utilities.

All functions here are API-compatible with librosa.
The hot-path computations run in Rust via _core.
"""
from __future__ import annotations

import numpy as np

from audiolib._core import (
    frames_to_samples as _frames_to_samples,
)
from audiolib._core import (
    frames_to_time as _frames_to_time,
)
from audiolib._core import (
    hz_to_mel as _hz_to_mel,
)
from audiolib._core import (
    hz_to_midi as _hz_to_midi,
)
from audiolib._core import (
    mel_to_hz as _mel_to_hz,
)
from audiolib._core import (
    midi_to_hz as _midi_to_hz,
)
from audiolib._core import (
    samples_to_frames as _samples_to_frames,
)
from audiolib._core import (
    samples_to_time as _samples_to_time,
)
from audiolib._core import (
    time_to_frames as _time_to_frames,
)
from audiolib._core import (
    time_to_samples as _time_to_samples,
)

__all__ = [
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
]

# ─── Note / pitch tables ──────────────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_NOTE_MAP = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3,
    "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7, "g#": 8,
    "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11,
}

# ─── Mel / Hz ─────────────────────────────────────────────────────────────────


def hz_to_mel(
    frequencies: float | np.ndarray | list,
    *,
    htk: bool = False,
) -> float | np.ndarray:
    """Convert Hz to Mels.

    API-compatible with ``librosa.hz_to_mel``.
    """
    scalar = np.isscalar(frequencies)
    arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    result = np.array(_hz_to_mel(arr.flatten().tolist(), htk), dtype=np.float64).reshape(arr.shape)
    return float(result.flat[0]) if scalar else result


def mel_to_hz(
    mels: float | np.ndarray | list,
    *,
    htk: bool = False,
) -> float | np.ndarray:
    """Convert Mels to Hz.

    API-compatible with ``librosa.mel_to_hz``.
    """
    scalar = np.isscalar(mels)
    arr = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    result = np.array(_mel_to_hz(arr.flatten().tolist(), htk), dtype=np.float64).reshape(arr.shape)
    return float(result.flat[0]) if scalar else result


# ─── MIDI / Hz ────────────────────────────────────────────────────────────────


def hz_to_midi(
    frequencies: float | np.ndarray | list,
) -> float | np.ndarray:
    """Convert Hz to MIDI note numbers.

    API-compatible with ``librosa.hz_to_midi``.
    """
    scalar = np.isscalar(frequencies)
    arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    result = np.array(_hz_to_midi(arr.flatten().tolist()), dtype=np.float64).reshape(arr.shape)
    return float(result.flat[0]) if scalar else result


def midi_to_hz(
    notes: float | np.ndarray | list,
) -> float | np.ndarray:
    """Convert MIDI note numbers to Hz.

    API-compatible with ``librosa.midi_to_hz``.
    """
    scalar = np.isscalar(notes)
    arr = np.atleast_1d(np.asarray(notes, dtype=np.float64))
    result = np.array(_midi_to_hz(arr.flatten().tolist()), dtype=np.float64).reshape(arr.shape)
    return float(result.flat[0]) if scalar else result


def midi_to_note(
    midi: int | float | np.ndarray | list,
    *,
    octave: bool = True,
    cents: bool = False,
) -> str | list:
    """Convert MIDI number(s) to note strings.

    API-compatible with ``librosa.midi_to_note``.
    """
    scalar = np.isscalar(midi)
    arr = np.atleast_1d(np.asarray(midi, dtype=np.float64))

    def _convert(m):
        pitch_class = int(round(m)) % 12
        note = _NOTE_NAMES[pitch_class]
        if octave:
            oct_num = int(round(m)) // 12 - 1
            note = f"{note}{oct_num}"
        if cents:
            cent_val = 100 * (m - round(m))
            note = f"{note}{cent_val:+.0f}"
        return note

    result = [_convert(m) for m in arr.flat]
    if scalar:
        return result[0]
    return np.array(result).reshape(arr.shape).tolist()


def note_to_midi(
    note: str | list,
    *,
    round_midi: bool = True,
) -> int | float | list:
    """Convert note string(s) to MIDI number(s).

    API-compatible with ``librosa.note_to_midi``.
    """
    def _convert(n):
        n = n.strip()
        # Extract note name (1 or 2 chars), then accidental, then octave
        i = 1
        if len(n) > 1 and n[1] in ("#", "b"):
            i = 2
        note_name = n[:i].lower()
        pitch = _NOTE_MAP.get(note_name, 0)
        rest = n[i:]
        try:
            octave = int(rest) if rest else 4
        except ValueError:
            octave = 4
        midi = (octave + 1) * 12 + pitch
        return int(midi) if round_midi else float(midi)

    if isinstance(note, str):
        return _convert(note)
    return [_convert(n) for n in note]


def note_to_hz(
    note: str | list,
    **kwargs,
) -> float | np.ndarray:
    """Convert note string(s) to Hz.

    API-compatible with ``librosa.note_to_hz``.
    """
    midi = note_to_midi(note, round_midi=False)
    if isinstance(midi, (int, float)):
        return float(midi_to_hz(float(midi)))
    return midi_to_hz(np.array(midi, dtype=np.float64))


def hz_to_note(
    frequencies: float | np.ndarray | list,
    **kwargs,
) -> str | list:
    """Convert Hz to nearest note name(s).

    API-compatible with ``librosa.hz_to_note``.
    """
    midi = hz_to_midi(frequencies)
    return midi_to_note(midi, **{k: v for k, v in kwargs.items() if k in ("octave", "cents")})


# ─── Frame / sample / time conversions ───────────────────────────────────────


def frames_to_samples(
    frames: int | np.ndarray | list,
    *,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> int | np.ndarray:
    """Convert frame indices to audio sample indices.

    API-compatible with ``librosa.frames_to_samples``.
    """
    scalar = np.isscalar(frames)
    arr = np.atleast_1d(np.asarray(frames, dtype=np.int64))
    result = np.array(_frames_to_samples(arr.flatten().tolist(), hop_length, n_fft), dtype=np.int64).reshape(arr.shape)
    return int(result.flat[0]) if scalar else result


def frames_to_time(
    frames: int | np.ndarray | list,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> float | np.ndarray:
    """Convert frame counts to time (seconds).

    API-compatible with ``librosa.frames_to_time``.
    """
    scalar = np.isscalar(frames)
    arr = np.atleast_1d(np.asarray(frames, dtype=np.int64))
    result = np.array(_frames_to_time(arr.flatten().tolist(), float(sr), hop_length, n_fft), dtype=np.float64).reshape(arr.shape)
    return float(result.flat[0]) if scalar else result


def samples_to_frames(
    samples: int | np.ndarray | list,
    *,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> int | np.ndarray:
    """Convert sample indices to STFT frames.

    API-compatible with ``librosa.samples_to_frames``.
    """
    scalar = np.isscalar(samples)
    arr = np.atleast_1d(np.asarray(samples, dtype=np.int64))
    result = np.array(_samples_to_frames(arr.flatten().tolist(), hop_length, n_fft), dtype=np.int64).reshape(arr.shape)
    return int(result.flat[0]) if scalar else result


def samples_to_time(
    samples: int | np.ndarray | list,
    *,
    sr: float = 22050,
) -> float | np.ndarray:
    """Convert sample indices to time (seconds).

    API-compatible with ``librosa.samples_to_time``.
    """
    scalar = np.isscalar(samples)
    arr = np.atleast_1d(np.asarray(samples, dtype=np.int64))
    result = np.array(_samples_to_time(arr.flatten().tolist(), float(sr)), dtype=np.float64).reshape(arr.shape)
    return float(result.flat[0]) if scalar else result


def time_to_frames(
    times: float | np.ndarray | list,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> int | np.ndarray:
    """Convert timestamps (seconds) to STFT frames.

    API-compatible with ``librosa.time_to_frames``.
    """
    scalar = np.isscalar(times)
    arr = np.atleast_1d(np.asarray(times, dtype=np.float64))
    result = np.array(_time_to_frames(arr.flatten().tolist(), float(sr), hop_length, n_fft), dtype=np.int64).reshape(arr.shape)
    return int(result.flat[0]) if scalar else result


def time_to_samples(
    times: float | np.ndarray | list,
    *,
    sr: float = 22050,
) -> int | np.ndarray:
    """Convert timestamps (seconds) to sample indices.

    API-compatible with ``librosa.time_to_samples``.
    """
    scalar = np.isscalar(times)
    arr = np.atleast_1d(np.asarray(times, dtype=np.float64))
    result = np.array(_time_to_samples(arr.flatten().tolist(), float(sr)), dtype=np.int64).reshape(arr.shape)
    return int(result.flat[0]) if scalar else result


# ─── Frequency range utilities ────────────────────────────────────────────────


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    """Alternative interface for np.fft.rfftfreq.

    API-compatible with ``librosa.fft_frequencies``.
    """
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel_frequencies(
    n_mels: int = 128,
    *,
    fmin: float = 0.0,
    fmax: float = 11025.0,
    htk: bool = False,
) -> np.ndarray:
    """Compute an array of acoustic frequencies tuned to the mel scale.

    API-compatible with ``librosa.mel_frequencies``.
    """
    min_mel = float(hz_to_mel(fmin, htk=htk))
    max_mel = float(hz_to_mel(fmax, htk=htk))
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)
