"""
Tests for audiolib.convert — frequency/unit conversion functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SR, HOP_LENGTH, N_FFT


# ---------------------------------------------------------------------------
# hz_to_mel / mel_to_hz
# ---------------------------------------------------------------------------

class TestHzMelRoundtrip:
    @pytest.mark.parametrize("hz", [0.0, 110.0, 440.0, 1000.0, 8000.0, 22050.0])
    def test_roundtrip_scalar(self, hz):
        from audiolib.convert import hz_to_mel, mel_to_hz
        mel = hz_to_mel(hz)
        hz_back = mel_to_hz(mel)
        assert abs(hz_back - hz) < 0.01, f"Roundtrip failed for {hz} Hz"

    def test_roundtrip_array(self):
        from audiolib.convert import hz_to_mel, mel_to_hz
        hz = np.array([0.0, 110.0, 440.0, 1000.0, 8000.0], dtype=np.float32)
        mel = hz_to_mel(hz)
        hz_back = mel_to_hz(mel)
        np.testing.assert_allclose(hz_back, hz, atol=0.02)

    def test_htk_roundtrip(self):
        from audiolib.convert import hz_to_mel, mel_to_hz
        hz = np.array([100.0, 440.0, 1000.0, 8000.0], dtype=np.float32)
        mel = hz_to_mel(hz, htk=True)
        hz_back = mel_to_hz(mel, htk=True)
        np.testing.assert_allclose(hz_back, hz, rtol=1e-4)

    def test_0hz_maps_to_0mel(self):
        from audiolib.convert import hz_to_mel
        assert hz_to_mel(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_monotone_increasing(self):
        from audiolib.convert import hz_to_mel
        hz = np.linspace(0, 8000, 100)
        mel = hz_to_mel(hz)
        assert (np.diff(mel) > 0).all()


# ---------------------------------------------------------------------------
# hz_to_midi / midi_to_hz
# ---------------------------------------------------------------------------

class TestHzMidiRoundtrip:
    @pytest.mark.parametrize("note,hz", [
        (69, 440.0),   # A4
        (60, 261.63),  # C4
        (57, 220.0),   # A3
    ])
    def test_midi_to_hz_known(self, note, hz):
        from audiolib.convert import midi_to_hz
        result = midi_to_hz(float(note))
        assert abs(result - hz) < 0.5, f"MIDI {note} → expected {hz}, got {result}"

    def test_hz_to_midi_A4(self):
        from audiolib.convert import hz_to_midi
        midi = hz_to_midi(440.0)
        assert abs(midi - 69.0) < 0.01

    def test_roundtrip_array(self):
        from audiolib.convert import hz_to_midi, midi_to_hz
        hz = np.array([220.0, 440.0, 880.0], dtype=np.float64)
        midi = hz_to_midi(hz)
        hz_back = midi_to_hz(midi)
        np.testing.assert_allclose(hz_back, hz, rtol=1e-5)


# ---------------------------------------------------------------------------
# note_to_midi / midi_to_note / note_to_hz / hz_to_note
# ---------------------------------------------------------------------------

class TestNoteConversions:
    @pytest.mark.parametrize("note,midi", [
        ("A4", 69),
        ("C4", 60),
        ("C#4", 61),
        ("Db4", 61),
        ("B3", 59),
    ])
    def test_note_to_midi(self, note, midi):
        from audiolib.convert import note_to_midi
        assert note_to_midi(note) == midi

    def test_midi_to_note_A4(self):
        from audiolib.convert import midi_to_note
        assert midi_to_note(69) == "A4"

    def test_note_roundtrip(self):
        from audiolib.convert import note_to_midi, midi_to_note
        for note in ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]:
            midi = note_to_midi(note)
            back = midi_to_note(midi)
            assert back == note

    def test_note_to_hz_A4(self):
        from audiolib.convert import note_to_hz
        hz = note_to_hz("A4")
        assert abs(hz - 440.0) < 0.01

    def test_hz_to_note_440(self):
        from audiolib.convert import hz_to_note
        note = hz_to_note(440.0)
        assert note == "A4"


# ---------------------------------------------------------------------------
# frames_to_samples / samples_to_frames
# ---------------------------------------------------------------------------

class TestFrameSampleConversions:
    def test_frames_to_samples_scalar(self):
        from audiolib.convert import frames_to_samples
        s = frames_to_samples(4, hop_length=HOP_LENGTH)
        assert s == 4 * HOP_LENGTH

    def test_frames_to_samples_array(self):
        from audiolib.convert import frames_to_samples
        frames = np.array([0, 1, 2, 4], dtype=np.int32)
        result = frames_to_samples(frames, hop_length=HOP_LENGTH)
        expected = frames * HOP_LENGTH
        np.testing.assert_array_equal(result, expected)

    def test_samples_to_frames_roundtrip(self):
        from audiolib.convert import frames_to_samples, samples_to_frames
        for f in [0, 1, 5, 10]:
            s = frames_to_samples(f, hop_length=HOP_LENGTH)
            f2 = samples_to_frames(s, hop_length=HOP_LENGTH)
            assert f2 == f


# ---------------------------------------------------------------------------
# frames_to_time / time_to_frames
# ---------------------------------------------------------------------------

class TestTimeConversions:
    def test_frames_to_time_scalar(self):
        from audiolib.convert import frames_to_time
        t = frames_to_time(10, hop_length=HOP_LENGTH, sr=SR)
        expected = 10 * HOP_LENGTH / SR
        assert abs(t - expected) < 1e-6

    def test_time_to_frames_roundtrip(self):
        from audiolib.convert import frames_to_time, time_to_frames
        for f in [0, 1, 5, 10]:
            t = frames_to_time(f, hop_length=HOP_LENGTH, sr=SR)
            f2 = time_to_frames(t, hop_length=HOP_LENGTH, sr=SR)
            assert f2 == f

    def test_time_to_frames_array(self):
        from audiolib.convert import time_to_frames
        times = np.array([0.0, 0.1, 0.5, 1.0])
        frames = time_to_frames(times, hop_length=HOP_LENGTH, sr=SR)
        assert frames.shape == times.shape


# ---------------------------------------------------------------------------
# samples_to_time / time_to_samples
# ---------------------------------------------------------------------------

class TestSamplesTimeConversions:
    def test_samples_to_time(self):
        from audiolib.convert import samples_to_time
        t = samples_to_time(SR, sr=SR)
        assert abs(t - 1.0) < 1e-6

    def test_time_to_samples(self):
        from audiolib.convert import time_to_samples
        s = time_to_samples(1.0, sr=SR)
        assert s == SR


# ---------------------------------------------------------------------------
# fft_frequencies / mel_frequencies
# ---------------------------------------------------------------------------

class TestFFTFrequencies:
    def test_shape(self):
        from audiolib.convert import fft_frequencies
        freqs = fft_frequencies(sr=SR, n_fft=N_FFT)
        assert len(freqs) == N_FFT // 2 + 1

    def test_first_bin_dc(self):
        from audiolib.convert import fft_frequencies
        freqs = fft_frequencies(sr=SR, n_fft=N_FFT)
        assert freqs[0] == pytest.approx(0.0)

    def test_last_bin_nyquist(self):
        from audiolib.convert import fft_frequencies
        freqs = fft_frequencies(sr=SR, n_fft=N_FFT)
        assert freqs[-1] == pytest.approx(SR / 2.0)


class TestMelFrequencies:
    def test_shape(self):
        from audiolib.convert import mel_frequencies
        freqs = mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
        assert len(freqs) == 128

    def test_monotone_increasing(self):
        from audiolib.convert import mel_frequencies
        freqs = mel_frequencies(n_mels=128)
        assert (np.diff(freqs) > 0).all()

    def test_fmin_fmax_respected(self):
        from audiolib.convert import mel_frequencies
        fmin, fmax = 80.0, 8000.0
        freqs = mel_frequencies(n_mels=128, fmin=fmin, fmax=fmax)
        assert freqs[0] >= fmin - 1.0
        assert freqs[-1] <= fmax + 1.0
