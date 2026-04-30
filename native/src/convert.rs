use pyo3::prelude::*;
use std::f64::consts::PI;

// ─── Mel / Hz conversion ─────────────────────────────────────────────────────

/// Convert Hz to Mels
#[pyfunction]
#[pyo3(signature = (frequencies, htk=false))]
pub fn hz_to_mel(frequencies: Vec<f64>, htk: bool) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&f| {
            if htk {
                2595.0 * (1.0 + f / 700.0).log10()
            } else {
                let f_min = 0.0f64;
                let f_sp = 200.0 / 3.0;
                let min_log_hz = 1000.0f64;
                let min_log_mel = (min_log_hz - f_min) / f_sp;
                let logstep = (6.4f64).ln() / 27.0;
                if f < min_log_hz {
                    (f - f_min) / f_sp
                } else {
                    min_log_mel + (f / min_log_hz).ln() / logstep
                }
            }
        })
        .collect()
}

/// Convert Mels to Hz
#[pyfunction]
#[pyo3(signature = (mels, htk=false))]
pub fn mel_to_hz(mels: Vec<f64>, htk: bool) -> Vec<f64> {
    mels.iter()
        .map(|&m| {
            if htk {
                700.0 * (10.0f64.powf(m / 2595.0) - 1.0)
            } else {
                let f_min = 0.0f64;
                let f_sp = 200.0 / 3.0;
                let min_log_hz = 1000.0f64;
                let min_log_mel = (min_log_hz - f_min) / f_sp;
                let logstep = (6.4f64).ln() / 27.0;
                if m < min_log_mel {
                    f_min + f_sp * m
                } else {
                    min_log_hz * (logstep * (m - min_log_mel)).exp()
                }
            }
        })
        .collect()
}

// ─── MIDI / Hz ────────────────────────────────────────────────────────────────

/// Convert Hz to MIDI note numbers
#[pyfunction]
pub fn hz_to_midi(frequencies: Vec<f64>) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&f| 12.0 * (f / 440.0).log2() + 69.0)
        .collect()
}

/// Convert MIDI note numbers to Hz
#[pyfunction]
pub fn midi_to_hz(notes: Vec<f64>) -> Vec<f64> {
    notes
        .iter()
        .map(|&n| 440.0 * 2.0f64.powf((n - 69.0) / 12.0))
        .collect()
}

// ─── Frame / Sample / Time conversions ───────────────────────────────────────

/// Convert frame indices to audio sample indices
#[pyfunction]
#[pyo3(signature = (frames, hop_length=512, n_fft=None))]
pub fn frames_to_samples(frames: Vec<i64>, hop_length: i64, n_fft: Option<i64>) -> Vec<i64> {
    let offset = n_fft.map(|n| n / 2).unwrap_or(0);
    frames.iter().map(|&f| f * hop_length + offset).collect()
}

/// Convert frame counts to time in seconds
#[pyfunction]
#[pyo3(signature = (frames, sr=22050.0, hop_length=512, n_fft=None))]
pub fn frames_to_time(frames: Vec<i64>, sr: f64, hop_length: i64, n_fft: Option<i64>) -> Vec<f64> {
    let samples = frames_to_samples(frames, hop_length, n_fft);
    samples.iter().map(|&s| s as f64 / sr).collect()
}

/// Convert sample indices to STFT frames
#[pyfunction]
#[pyo3(signature = (samples, hop_length=512, n_fft=None))]
pub fn samples_to_frames(samples: Vec<i64>, hop_length: i64, n_fft: Option<i64>) -> Vec<i64> {
    let offset = n_fft.map(|n| n / 2).unwrap_or(0);
    samples.iter().map(|&s| (s - offset) / hop_length).collect()
}

/// Convert sample indices to time in seconds
#[pyfunction]
#[pyo3(signature = (samples, sr=22050.0))]
pub fn samples_to_time(samples: Vec<i64>, sr: f64) -> Vec<f64> {
    samples.iter().map(|&s| s as f64 / sr).collect()
}

/// Convert timestamps (seconds) to STFT frames
#[pyfunction]
#[pyo3(signature = (times, sr=22050.0, hop_length=512, n_fft=None))]
pub fn time_to_frames(times: Vec<f64>, sr: f64, hop_length: i64, n_fft: Option<i64>) -> Vec<i64> {
    let samples: Vec<i64> = times.iter().map(|&t| (t * sr).round() as i64).collect();
    samples_to_frames(samples, hop_length, n_fft)
}

/// Convert timestamps (seconds) to sample indices
#[pyfunction]
#[pyo3(signature = (times, sr=22050.0))]
pub fn time_to_samples(times: Vec<f64>, sr: f64) -> Vec<i64> {
    times.iter().map(|&t| (t * sr).round() as i64).collect()
}

// Suppress unused import warning
#[allow(dead_code)]
const _PI_CHECK: f64 = PI;
