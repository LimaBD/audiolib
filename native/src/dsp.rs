use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::f64::consts::PI as PI64;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ─── STFT ─────────────────────────────────────────────────────────────────────

/// Short-time Fourier transform (STFT).
/// Returns interleaved [real, imag] values in a flat Vec<f32> with shape
/// (n_frames, n_fft/2 + 1, 2).
///
/// Parameters
/// ----------
/// y          : audio signal (mono, f32)
/// n_fft      : FFT window size (default 2048)
/// hop_length : number of samples between frames (default n_fft // 4)
/// win_length : window length in samples (default = n_fft)
/// center     : if true, pad signal at both ends (default true)
///
/// Returns
/// -------
/// flat Vec<f32> with shape info embedded; caller unpacks via Python layer
#[pyfunction]
#[pyo3(signature = (y, n_fft=2048, hop_length=None, win_length=None, center=true))]
pub fn stft(
    y: Vec<f32>,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    center: bool,
) -> PyResult<(Vec<f32>, usize, usize)> {
    let hop = hop_length.unwrap_or(n_fft / 4);
    let win_len = win_length.unwrap_or(n_fft);

    if n_fft == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_fft must be > 0",
        ));
    }
    if hop == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hop_length must be > 0",
        ));
    }

    let window = hann_window(win_len);

    // Pad signal if center=True
    let signal: Vec<f32> = if center {
        let pad = n_fft / 2;
        let mut padded = vec![0.0f32; pad + y.len() + pad];
        padded[pad..pad + y.len()].copy_from_slice(&y);
        padded
    } else {
        y
    };

    let n_bins = n_fft / 2 + 1;
    let n_frames = if signal.len() < n_fft {
        0
    } else {
        1 + (signal.len() - n_fft) / hop
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Result: [real, imag] interleaved, shape (n_frames, n_bins, 2)
    let mut out = Vec::with_capacity(n_frames * n_bins * 2);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let mut buf: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let sample = if start + i < signal.len() {
                    signal[start + i]
                } else {
                    0.0
                };
                let w = if i < win_len { window[i] } else { 0.0 };
                Complex::new(sample * w, 0.0)
            })
            .collect();

        fft.process(&mut buf);

        for item in buf.iter().take(n_bins) {
            out.push(item.re);
            out.push(item.im);
        }
    }

    Ok((out, n_frames, n_bins))
}

// ─── ISTFT ────────────────────────────────────────────────────────────────────

/// Inverse short-time Fourier transform (ISTFT).
///
/// Parameters
/// ----------
/// stft_re    : real parts, flat, shape (n_frames, n_bins)
/// stft_im    : imaginary parts, flat, shape (n_frames, n_bins)
/// n_frames   : number of frames
/// n_bins     : n_fft / 2 + 1
/// hop_length : samples between frames
/// win_length : window length
/// center     : whether STFT was centered
///
/// Returns
/// -------
/// reconstructed time-domain signal
#[pyfunction]
#[pyo3(signature = (stft_re, stft_im, n_frames, n_bins, hop_length=512, win_length=2048, center=true))]
pub fn istft(
    stft_re: Vec<f32>,
    stft_im: Vec<f32>,
    n_frames: usize,
    n_bins: usize,
    hop_length: usize,
    win_length: usize,
    center: bool,
) -> PyResult<Vec<f32>> {
    let n_fft = (n_bins - 1) * 2;
    let window = hann_window(win_length);

    let expected_len = if n_frames == 0 {
        0
    } else {
        (n_frames - 1) * hop_length + n_fft
    };

    let mut signal = vec![0.0f32; expected_len];
    let mut window_sum = vec![0.0f32; expected_len];

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    for frame_idx in 0..n_frames {
        let base = frame_idx * n_bins;
        let mut buf: Vec<Complex<f32>> = (0..n_bins)
            .map(|k| Complex::new(stft_re[base + k], stft_im[base + k]))
            .collect();

        // Mirror for real-valued IFFT
        for k in 1..(n_fft - n_bins + 1) {
            let mirror = Complex::new(
                stft_re[base + n_bins - 1 - k],
                -stft_im[base + n_bins - 1 - k],
            );
            buf.push(mirror);
        }
        buf.resize(n_fft, Complex::new(0.0, 0.0));

        ifft.process(&mut buf);

        let start = frame_idx * hop_length;
        for i in 0..n_fft {
            if start + i < signal.len() {
                let w = if i < win_length { window[i] } else { 0.0 };
                signal[start + i] += buf[i].re * w / n_fft as f32;
                window_sum[start + i] += w * w;
            }
        }
    }

    // Normalize
    for i in 0..signal.len() {
        if window_sum[i] > 1e-8 {
            signal[i] /= window_sum[i];
        }
    }

    // Trim center padding
    if center {
        let pad = n_fft / 2;
        if signal.len() > 2 * pad {
            return Ok(signal[pad..signal.len() - pad].to_vec());
        }
    }

    Ok(signal)
}

// ─── Magnitude scaling ────────────────────────────────────────────────────────

/// Convert amplitude spectrogram to dB
#[pyfunction]
#[pyo3(signature = (s, ref_val=1.0, amin=1e-5, top_db=None))]
pub fn amplitude_to_db(s: Vec<f32>, ref_val: f32, amin: f32, top_db: Option<f32>) -> Vec<f32> {
    let ref_db = 20.0 * ref_val.abs().log10();
    let mut out: Vec<f32> = s
        .iter()
        .map(|&x| {
            let a = x.abs().max(amin);
            20.0 * a.log10() - ref_db
        })
        .collect();
    if let Some(top) = top_db {
        let max_val = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let floor = max_val - top;
        out.iter_mut().for_each(|v| {
            if *v < floor {
                *v = floor;
            }
        });
    }
    out
}

/// Convert power spectrogram to dB
#[pyfunction]
#[pyo3(signature = (s, ref_val=1.0, amin=1e-10, top_db=None))]
pub fn power_to_db(s: Vec<f32>, ref_val: f32, amin: f32, top_db: Option<f32>) -> Vec<f32> {
    let ref_db = 10.0 * ref_val.abs().log10();
    let mut out: Vec<f32> = s
        .iter()
        .map(|&x| {
            let a = x.abs().max(amin);
            10.0 * a.log10() - ref_db
        })
        .collect();
    if let Some(top) = top_db {
        let max_val = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let floor = max_val - top;
        out.iter_mut().for_each(|v| {
            if *v < floor {
                *v = floor;
            }
        });
    }
    out
}

/// Convert dB-scaled spectrogram to amplitude
#[pyfunction]
#[pyo3(signature = (s_db, ref_val=1.0))]
pub fn db_to_amplitude(s_db: Vec<f32>, ref_val: f32) -> Vec<f32> {
    s_db.iter()
        .map(|&x| ref_val * 10.0f32.powf(x / 20.0))
        .collect()
}

/// Convert dB-scale to power
#[pyfunction]
#[pyo3(signature = (s_db, ref_val=1.0))]
pub fn db_to_power(s_db: Vec<f32>, ref_val: f32) -> Vec<f32> {
    s_db.iter()
        .map(|&x| ref_val * 10.0f32.powf(x / 10.0))
        .collect()
}

// ─── Time-domain processing ───────────────────────────────────────────────────

/// Find the zero-crossings of a signal
#[pyfunction]
#[pyo3(signature = (y, threshold=1e-10, pad=true))]
pub fn zero_crossings(y: Vec<f32>, threshold: f32, pad: bool) -> Vec<bool> {
    let _ = pad;
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    let mut zc = vec![false; n];
    for i in 1..n {
        let a = if y[i - 1].abs() <= threshold {
            0.0
        } else {
            y[i - 1]
        };
        let b = if y[i].abs() <= threshold { 0.0 } else { y[i] };
        zc[i] = a.signum() != b.signum();
    }
    zc
}

/// Bounded-lag auto-correlation
#[pyfunction]
#[pyo3(signature = (y, max_size=None))]
pub fn autocorrelate(y: Vec<f32>, max_size: Option<usize>) -> Vec<f32> {
    let n = y.len();
    let limit = max_size.unwrap_or(n).min(n);
    let fft_size = next_power_of_two(2 * n - 1);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut buf: Vec<Complex<f32>> = y.iter().map(|&x| Complex::new(x, 0.0)).collect();
    buf.resize(fft_size, Complex::new(0.0, 0.0));

    fft.process(&mut buf);

    // Power spectrum
    buf.iter_mut().for_each(|c| {
        let power = c.re * c.re + c.im * c.im;
        *c = Complex::new(power, 0.0);
    });

    ifft.process(&mut buf);

    let scale = fft_size as f32;
    buf[..limit].iter().map(|c| c.re / scale).collect()
}

/// mu-law compression
#[pyfunction]
#[pyo3(signature = (x, mu=255.0, quantize=false))]
pub fn mu_compress(x: Vec<f32>, mu: f32, quantize: bool) -> PyResult<Vec<f32>> {
    if mu <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mu must be strictly positive",
        ));
    }
    let compressed: Vec<f32> = x
        .iter()
        .map(|&v| {
            let s = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
            s * (1.0 + mu * v.abs()).ln() / (1.0 + mu).ln()
        })
        .collect();

    if quantize {
        let half = ((mu + 1.0) / 2.0) as i32;
        Ok(compressed
            .iter()
            .map(|&v| {
                let idx = ((v + 1.0) / 2.0 * (mu + 1.0)) as i32;
                (idx - half) as f32
            })
            .collect())
    } else {
        Ok(compressed)
    }
}

/// mu-law expansion
#[pyfunction]
#[pyo3(signature = (x, mu=255.0, quantize=false))]
pub fn mu_expand(x: Vec<f32>, mu: f32, quantize: bool) -> PyResult<Vec<f32>> {
    if mu <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mu must be strictly positive",
        ));
    }
    let vals: Vec<f32> = if quantize {
        x.iter().map(|&v| v * 2.0 / (1.0 + mu)).collect()
    } else {
        x.clone()
    };
    Ok(vals
        .iter()
        .map(|&v| {
            let s = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
            s / mu * ((1.0 + mu).powf(v.abs()) - 1.0)
        })
        .collect())
}

// ─── Audio utilities ──────────────────────────────────────────────────────────

/// Convert multi-channel audio to mono by averaging channels.
/// y_flat: flat array, n_channels interleaved (for 2ch: L0 R0 L1 R1 ...)
/// Actually we expect channel-first: first n_samples for ch0, next for ch1...
/// n_channels: number of channels
#[pyfunction]
#[allow(unknown_lints, clippy::manual_is_multiple_of)]
pub fn to_mono(y_flat: Vec<f32>, n_channels: usize) -> PyResult<Vec<f32>> {
    if n_channels == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_channels must be > 0",
        ));
    }
    if y_flat.len() % n_channels != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_flat length must be divisible by n_channels",
        ));
    }
    let n_samples = y_flat.len() / n_channels;
    let inv = 1.0 / n_channels as f32;
    let out: Vec<f32> = (0..n_samples)
        .map(|i| {
            let sum: f32 = (0..n_channels).map(|c| y_flat[c * n_samples + i]).sum();
            sum * inv
        })
        .collect();
    Ok(out)
}

/// Linear resample using simple linear interpolation.
/// For high-quality resampling users should use the Python layer which calls soundfile/soxr.
#[pyfunction]
pub fn resample(y: Vec<f32>, orig_sr: f32, target_sr: f32) -> PyResult<Vec<f32>> {
    if orig_sr <= 0.0 || target_sr <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sampling rates must be positive",
        ));
    }
    if (orig_sr - target_sr).abs() < 1e-6 {
        return Ok(y);
    }
    let ratio = target_sr / orig_sr;
    let n_out = ((y.len() as f32 * ratio).ceil()) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let src = i as f32 / ratio;
        let idx = src as usize;
        let frac = src - idx as f32;
        let a = *y.get(idx).unwrap_or(&0.0);
        let b = *y.get(idx + 1).unwrap_or(&0.0);
        out.push(a + frac * (b - a));
    }
    Ok(out)
}

/// Root mean square energy
#[pyfunction]
#[pyo3(signature = (y, frame_length=2048, hop_length=512))]
pub fn get_rms(y: Vec<f32>, frame_length: usize, hop_length: usize) -> Vec<f32> {
    let n = y.len();
    if n == 0 || frame_length == 0 || hop_length == 0 {
        return vec![];
    }
    let n_frames = if n < frame_length {
        0
    } else {
        1 + (n - frame_length) / hop_length
    };
    let inv = 1.0 / frame_length as f32;
    (0..n_frames)
        .map(|i| {
            let start = i * hop_length;
            let sum_sq: f32 = y[start..start + frame_length].iter().map(|&v| v * v).sum();
            (sum_sq * inv).sqrt()
        })
        .collect()
}

// ─── Mel filterbank ───────────────────────────────────────────────────────────

/// Build a triangular mel filterbank.
/// Returns flat Vec<f32> with shape (n_mels, n_fft/2 + 1).
#[pyfunction]
#[pyo3(signature = (sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=false))]
pub fn mel_filterbank(
    sr: f32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: Option<f32>,
    htk: bool,
) -> Vec<f32> {
    let fmax = fmax.unwrap_or(sr / 2.0);
    let n_bins = n_fft / 2 + 1;

    let hz_to_mel = |f: f32| -> f32 {
        if htk {
            2595.0 * (1.0 + f / 700.0).log10()
        } else {
            let f_min = 0.0f32;
            let f_sp = 200.0f32 / 3.0;
            let min_log_hz = 1000.0f32;
            let min_log_mel = (min_log_hz - f_min) / f_sp;
            let logstep = (6.4f32).ln() / 27.0;
            if f < min_log_hz {
                (f - f_min) / f_sp
            } else {
                min_log_mel + (f / min_log_hz).ln() / logstep
            }
        }
    };

    let mel_to_hz = |m: f32| -> f32 {
        if htk {
            700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
        } else {
            let f_min = 0.0f32;
            let f_sp = 200.0f32 / 3.0;
            let min_log_hz = 1000.0f32;
            let min_log_mel = (min_log_hz - f_min) / f_sp;
            let logstep = (6.4f32).ln() / 27.0;
            if m < min_log_mel {
                f_min + f_sp * m
            } else {
                min_log_hz * (logstep * (m - min_log_mel)).exp()
            }
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // mel center points (n_mels + 2 uniformly spaced in mel)
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // FFT frequencies in Hz
    let fft_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();

    let mut weights = vec![0.0f32; n_mels * n_bins];

    for m in 0..n_mels {
        let lower = mel_to_hz(mel_points[m]);
        let center = mel_to_hz(mel_points[m + 1]);
        let upper = mel_to_hz(mel_points[m + 2]);
        let lower_width = center - lower;
        let upper_width = upper - center;

        for k in 0..n_bins {
            let f = fft_freqs[k];
            let w = if f >= lower && f <= center && lower_width > 0.0 {
                (f - lower) / lower_width
            } else if f > center && f <= upper && upper_width > 0.0 {
                (upper - f) / upper_width
            } else {
                0.0
            };
            weights[m * n_bins + k] = w;
        }
    }

    weights
}

/// Compute a mel spectrogram from an STFT magnitude.
/// stft_mag: flat f32 Vec, shape (n_frames, n_bins)
/// mel_fb: flat f32 Vec, shape (n_mels, n_bins)
/// Returns flat f32 Vec, shape (n_mels, n_frames)
#[pyfunction]
pub fn melspectrogram(
    stft_mag: Vec<f32>,
    mel_fb: Vec<f32>,
    n_frames: usize,
    n_bins: usize,
    n_mels: usize,
) -> Vec<f32> {
    // Output: (n_mels, n_frames)
    let mut out = vec![0.0f32; n_mels * n_frames];
    for t in 0..n_frames {
        for m in 0..n_mels {
            let mut acc = 0.0f32;
            for k in 0..n_bins {
                acc += mel_fb[m * n_bins + k] * stft_mag[t * n_bins + k];
            }
            out[m * n_frames + t] = acc;
        }
    }
    out
}

/// Compute MFCCs from log mel spectrogram using DCT-II.
/// log_mel: flat f32, shape (n_mels, n_frames)
/// n_mfcc: number of coefficients to return
/// Returns flat f32, shape (n_mfcc, n_frames)
#[pyfunction]
pub fn mfcc(log_mel: Vec<f32>, n_mels: usize, n_frames: usize, n_mfcc: usize) -> Vec<f32> {
    let n_mfcc = n_mfcc.min(n_mels);
    let mut out = vec![0.0f32; n_mfcc * n_frames];
    // DCT-II: c[k,t] = sum_{m=0}^{n_mels-1} log_mel[m,t] * cos(pi*k*(m+0.5)/n_mels)
    for t in 0..n_frames {
        for k in 0..n_mfcc {
            let mut acc = 0.0f32;
            for m in 0..n_mels {
                acc += log_mel[m * n_frames + t]
                    * (PI * k as f32 * (m as f32 + 0.5) / n_mels as f32).cos();
            }
            out[k * n_frames + t] = acc;
        }
    }
    out
}

// ─── Chroma ───────────────────────────────────────────────────────────────────

/// Compute chromagram from STFT magnitude.
/// stft_mag: flat f32, shape (n_frames, n_bins)
/// Returns flat f32, shape (12, n_frames)
#[pyfunction]
#[pyo3(signature = (stft_mag, sr, n_fft, n_frames, n_bins, tuning=0.0))]
pub fn chroma_stft(
    stft_mag: Vec<f32>,
    sr: f32,
    n_fft: usize,
    n_frames: usize,
    n_bins: usize,
    tuning: f32,
) -> Vec<f32> {
    let a4 = 440.0f32 * 2.0f32.powf(tuning / 12.0);
    let n_chroma = 12usize;
    let mut out = vec![0.0f32; n_chroma * n_frames];

    for k in 1..n_bins {
        let freq = k as f32 * sr / n_fft as f32;
        let midi_float = 12.0 * (freq / a4).log2() + 69.0;
        let chroma_idx =
            ((midi_float.round() as i64 % n_chroma as i64 + n_chroma as i64) as usize) % n_chroma;
        for t in 0..n_frames {
            out[chroma_idx * n_frames + t] += stft_mag[t * n_bins + k];
        }
    }

    // L2 normalize each frame
    for t in 0..n_frames {
        let norm: f32 = (0..n_chroma)
            .map(|c| out[c * n_frames + t].powi(2))
            .sum::<f32>()
            .sqrt();
        if norm > 1e-8 {
            for c in 0..n_chroma {
                out[c * n_frames + t] /= norm;
            }
        }
    }

    out
}

// ─── Spectral features ────────────────────────────────────────────────────────

/// Spectral centroid
#[pyfunction]
pub fn spectral_centroid(
    stft_mag: Vec<f32>,
    sr: f32,
    n_fft: usize,
    n_frames: usize,
    n_bins: usize,
) -> Vec<f32> {
    let freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();
    (0..n_frames)
        .map(|t| {
            let mag_sum: f32 = (0..n_bins).map(|k| stft_mag[t * n_bins + k]).sum();
            if mag_sum < 1e-10 {
                return 0.0;
            }
            (0..n_bins)
                .map(|k| freqs[k] * stft_mag[t * n_bins + k])
                .sum::<f32>()
                / mag_sum
        })
        .collect()
}

/// Spectral bandwidth
#[pyfunction]
#[pyo3(signature = (stft_mag, sr, n_fft, n_frames, n_bins, p=2.0))]
pub fn spectral_bandwidth(
    stft_mag: Vec<f32>,
    sr: f32,
    n_fft: usize,
    n_frames: usize,
    n_bins: usize,
    p: f32,
) -> Vec<f32> {
    let freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();
    let centroids = spectral_centroid(stft_mag.clone(), sr, n_fft, n_frames, n_bins);
    (0..n_frames)
        .map(|t| {
            let mag_sum: f32 = (0..n_bins).map(|k| stft_mag[t * n_bins + k]).sum();
            if mag_sum < 1e-10 {
                return 0.0;
            }
            let c = centroids[t];
            let bw: f32 = (0..n_bins)
                .map(|k| stft_mag[t * n_bins + k] * (freqs[k] - c).abs().powf(p))
                .sum::<f32>()
                / mag_sum;
            bw.powf(1.0 / p)
        })
        .collect()
}

/// Spectral rolloff
#[pyfunction]
#[pyo3(signature = (stft_mag, sr, n_fft, n_frames, n_bins, roll_percent=0.85))]
pub fn spectral_rolloff(
    stft_mag: Vec<f32>,
    sr: f32,
    n_fft: usize,
    n_frames: usize,
    n_bins: usize,
    roll_percent: f32,
) -> Vec<f32> {
    let freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();
    (0..n_frames)
        .map(|t| {
            let total: f32 = (0..n_bins).map(|k| stft_mag[t * n_bins + k]).sum();
            let threshold = roll_percent * total;
            let mut cumsum = 0.0f32;
            for k in 0..n_bins {
                cumsum += stft_mag[t * n_bins + k];
                if cumsum >= threshold {
                    return freqs[k];
                }
            }
            freqs[n_bins - 1]
        })
        .collect()
}

/// Spectral flatness
#[pyfunction]
pub fn spectral_flatness(stft_mag: Vec<f32>, n_frames: usize, n_bins: usize) -> Vec<f32> {
    (0..n_frames)
        .map(|t| {
            let frame = &stft_mag[t * n_bins..(t + 1) * n_bins];
            let geo_mean = {
                let log_sum: f64 = frame.iter().map(|&v| (v as f64 + 1e-10).ln()).sum();
                (log_sum / n_bins as f64).exp() as f32
            };
            let arith_mean: f32 = frame.iter().sum::<f32>() / n_bins as f32;
            if arith_mean < 1e-10 {
                0.0
            } else {
                geo_mean / arith_mean
            }
        })
        .collect()
}

// ─── Onset strength ───────────────────────────────────────────────────────────

/// Simple onset strength (spectral flux)
#[pyfunction]
pub fn onset_strength(stft_mag: Vec<f32>, n_frames: usize, n_bins: usize) -> Vec<f32> {
    if n_frames == 0 {
        return vec![];
    }
    let mut odf = vec![0.0f32; n_frames];
    for t in 1..n_frames {
        let flux: f32 = (0..n_bins)
            .map(|k| {
                let diff = stft_mag[t * n_bins + k] - stft_mag[(t - 1) * n_bins + k];
                diff.max(0.0)
            })
            .sum();
        odf[t] = flux;
    }
    odf
}

// ─── YIN pitch estimation ─────────────────────────────────────────────────────

/// YIN fundamental frequency estimation.
/// Returns Vec<f32> of f0 estimates per frame (0.0 = unvoiced).
#[pyfunction]
#[pyo3(signature = (y, fmin, fmax, sr=22050.0, frame_length=2048, hop_length=None, trough_threshold=0.1))]
pub fn yin(
    y: Vec<f32>,
    fmin: f32,
    fmax: f32,
    sr: f32,
    frame_length: usize,
    hop_length: Option<usize>,
    trough_threshold: f32,
) -> PyResult<Vec<f32>> {
    if fmin <= 0.0 || fmax <= 0.0 || fmax <= fmin {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fmin and fmax must be positive with fmax > fmin",
        ));
    }
    let hop = hop_length.unwrap_or(frame_length / 4);
    let tau_min = (sr / fmax).floor() as usize;
    let tau_max = ((sr / fmin).ceil() as usize).min(frame_length / 2);

    if tau_min >= tau_max || tau_max == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fmin/fmax/frame_length combination is invalid",
        ));
    }

    let n_frames = if y.len() < frame_length {
        0
    } else {
        1 + (y.len() - frame_length) / hop
    };

    let mut f0 = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let frame = &y[start..start + frame_length];

        // If the frame is essentially silent, mark as unvoiced
        let frame_energy: f32 = frame.iter().map(|&v| v * v).sum::<f32>() / frame_length as f32;
        if frame_energy < 1e-10 {
            f0.push(0.0);
            continue;
        }

        // Difference function d(tau)
        let mut diff = vec![0.0f32; tau_max + 1];
        for tau in 1..=tau_max {
            let mut s = 0.0f32;
            for j in 0..(frame_length - tau) {
                let v = frame[j] - frame[j + tau];
                s += v * v;
            }
            diff[tau] = s;
        }

        // Cumulative mean normalized difference (CMND)
        let mut cmnd = vec![1.0f32; tau_max + 1];
        let mut running = 0.0f32;
        for tau in 1..=tau_max {
            running += diff[tau];
            cmnd[tau] = if running < 1e-10 {
                0.0
            } else {
                diff[tau] * tau as f32 / running
            };
        }

        // Find first trough below threshold in [tau_min, tau_max]
        let mut found = false;
        for tau in tau_min..tau_max {
            if cmnd[tau] < trough_threshold
                && (tau == tau_min || cmnd[tau] <= cmnd[tau - 1])
                && (tau + 1 > tau_max || cmnd[tau] <= cmnd[tau + 1])
            {
                f0.push(sr / tau as f32);
                found = true;
                break;
            }
        }
        if !found {
            // Find global minimum
            let best_tau = (tau_min..=tau_max)
                .min_by(|&a, &b| cmnd[a].partial_cmp(&cmnd[b]).unwrap())
                .unwrap_or(tau_min);
            if cmnd[best_tau] < trough_threshold {
                f0.push(sr / best_tau as f32);
            } else {
                f0.push(0.0);
            }
        }
    }

    Ok(f0)
}

// ─── Phase vocoder ────────────────────────────────────────────────────────────

/// Time-scale a complex STFT matrix using the phase vocoder algorithm.
///
/// Parameters
/// ----------
/// stft_re    : real parts flat, shape (n_bins, n_frames) row-major
/// stft_im    : imaginary parts flat, shape (n_bins, n_frames) row-major
/// n_bins     : number of frequency bins
/// n_frames   : number of input frames
/// rate       : time-stretch factor (>1 = speed up, <1 = slow down)
/// hop_length : hop length used in the original STFT
///
/// Returns
/// -------
/// (re_out, im_out, n_frames_out)
#[pyfunction]
#[pyo3(signature = (stft_re, stft_im, n_bins, n_frames, rate, hop_length=512))]
pub fn phase_vocoder(
    stft_re: Vec<f32>,
    stft_im: Vec<f32>,
    n_bins: usize,
    n_frames: usize,
    rate: f32,
    hop_length: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, usize)> {
    if rate <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "rate must be positive",
        ));
    }

    let n_frames_out = (n_frames as f32 / rate).round() as usize;
    if n_frames_out == 0 {
        return Ok((vec![], vec![], 0));
    }

    let mut re_out = vec![0.0f32; n_bins * n_frames_out];
    let mut im_out = vec![0.0f32; n_bins * n_frames_out];

    // Phase accumulator per bin (starts from first frame's phase)
    let mut phase_acc: Vec<f32> = (0..n_bins)
        .map(|k| {
            if n_frames > 0 {
                stft_im[k].atan2(stft_re[k])
            } else {
                0.0
            }
        })
        .collect();

    let omega: Vec<f32> = (0..n_bins)
        .map(|k| 2.0 * PI * k as f32 * hop_length as f32 / (2 * (n_bins - 1)) as f32)
        .collect();

    for out_t in 0..n_frames_out {
        // Corresponding input position (float)
        let in_t_f = out_t as f32 * rate;
        let in_t0 = in_t_f as usize;
        let in_t1 = (in_t0 + 1).min(n_frames - 1);
        let alpha = in_t_f - in_t0 as f32;

        for k in 0..n_bins {
            // Magnitude: linear interpolation
            let re0 = stft_re[in_t0 * n_bins + k];
            let im0 = stft_im[in_t0 * n_bins + k];
            let re1 = stft_re[in_t1 * n_bins + k];
            let im1 = stft_im[in_t1 * n_bins + k];

            let mag0 = (re0 * re0 + im0 * im0).sqrt();
            let mag1 = (re1 * re1 + im1 * im1).sqrt();
            let mag = (1.0 - alpha) * mag0 + alpha * mag1;

            // Phase advance: expected advance per step
            if out_t == 0 {
                let ph = stft_im[k].atan2(stft_re[k]);
                re_out[k] = mag * ph.cos();
                im_out[k] = mag * ph.sin();
            } else {
                // Phase gradient from previous input frame
                let ph0 = stft_im[in_t0 * n_bins + k].atan2(stft_re[in_t0 * n_bins + k]);
                let ph_prev = if in_t0 > 0 {
                    stft_im[(in_t0 - 1) * n_bins + k].atan2(stft_re[(in_t0 - 1) * n_bins + k])
                } else {
                    ph0 - omega[k]
                };
                let dp = ph0 - ph_prev - omega[k];
                // Wrap to [-π, π]
                let dp_wrapped = dp - 2.0 * PI * (dp / (2.0 * PI)).round();
                let true_dp = omega[k] + dp_wrapped;
                phase_acc[k] += true_dp * rate;
                let ph = phase_acc[k];
                re_out[out_t * n_bins + k] = mag * ph.cos();
                im_out[out_t * n_bins + k] = mag * ph.sin();
            }
        }
    }

    Ok((re_out, im_out, n_frames_out))
}

// ─── Griffin-Lim ─────────────────────────────────────────────────────────────

/// Reconstruct a signal from a magnitude STFT using the Griffin-Lim algorithm.
///
/// Parameters
/// ----------
/// S_flat     : magnitude spectrogram, flat Vec<f32>, shape (n_bins, n_frames) row-major
/// n_bins     : n_fft/2 + 1
/// n_frames   : number of frames
/// n_iter     : number of Griffin-Lim iterations (default 32)
/// hop_length : hop length
/// win_length : window length (default = n_fft = 2*(n_bins-1))
/// center     : whether the STFT was centered
///
/// Returns reconstructed signal as Vec<f32>
#[pyfunction]
#[pyo3(signature = (s_flat, n_bins, n_frames, n_iter=32, hop_length=512, win_length=None, center=true))]
pub fn griffinlim(
    s_flat: Vec<f32>,
    n_bins: usize,
    n_frames: usize,
    n_iter: usize,
    hop_length: usize,
    win_length: Option<usize>,
    center: bool,
) -> Vec<f32> {
    let n_fft = (n_bins - 1) * 2;
    let wl = win_length.unwrap_or(n_fft);
    let window = hann_window(wl);

    // Initialize with random phase
    let mut phase: Vec<f32> = (0..n_bins * n_frames)
        .map(|i| {
            // Deterministic pseudo-random from index
            let seed = (i * 1664525 + 1013904223) & 0x7FFFFFFF;
            (seed as f32 / 0x7FFFFFFF as f32) * 2.0 * PI - PI
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    let fft = planner.plan_fft_forward(n_fft);

    let mut signal = vec![
        0.0f32;
        if n_frames == 0 {
            0
        } else {
            (n_frames - 1) * hop_length + n_fft
        }
    ];

    for _iter in 0..n_iter {
        // Build complex STFT from magnitude + current phase
        // ISTFT → signal
        let n_sig = if n_frames == 0 {
            0
        } else {
            (n_frames - 1) * hop_length + n_fft
        };
        signal = vec![0.0f32; n_sig];
        let mut win_sum = vec![0.0f32; n_sig];

        for t in 0..n_frames {
            let mut buf: Vec<Complex<f32>> = (0..n_bins)
                .map(|k| {
                    let mag = s_flat[k * n_frames + t];
                    let ph = phase[k * n_frames + t];
                    Complex::new(mag * ph.cos(), mag * ph.sin())
                })
                .collect();

            // Mirror for real-valued IFFT
            for k in 1..(n_fft - n_bins + 1) {
                let mirror = Complex::new(buf[n_bins - 1 - k].re, -buf[n_bins - 1 - k].im);
                buf.push(mirror);
            }
            buf.resize(n_fft, Complex::new(0.0, 0.0));

            ifft.process(&mut buf);

            let start = t * hop_length;
            for i in 0..n_fft {
                if start + i < n_sig {
                    let w = if i < wl { window[i] } else { 0.0 };
                    signal[start + i] += buf[i].re * w / n_fft as f32;
                    win_sum[start + i] += w * w;
                }
            }
        }

        // Normalize by window sum
        for i in 0..n_sig {
            if win_sum[i] > 1e-8 {
                signal[i] /= win_sum[i];
            }
        }

        // STFT of current signal to update phases
        let analysis_signal: Vec<f32> = if center {
            let pad = n_fft / 2;
            let mut padded = vec![0.0f32; pad + signal.len() + pad];
            padded[pad..pad + signal.len()].copy_from_slice(&signal);
            padded
        } else {
            signal.clone()
        };

        for t in 0..n_frames {
            let start = t * hop_length;
            let mut buf: Vec<Complex<f32>> = (0..n_fft)
                .map(|i| {
                    let s = if start + i < analysis_signal.len() {
                        analysis_signal[start + i]
                    } else {
                        0.0
                    };
                    let w = if i < wl { window[i] } else { 0.0 };
                    Complex::new(s * w, 0.0)
                })
                .collect();
            fft.process(&mut buf);
            for k in 0..n_bins {
                phase[k * n_frames + t] = buf[k].im.atan2(buf[k].re);
            }
        }
    }

    // Trim center padding
    if center && !signal.is_empty() {
        let pad = n_fft / 2;
        if signal.len() > 2 * pad {
            return signal[pad..signal.len() - pad].to_vec();
        }
    }

    signal
}

// ─── Spectral contrast ────────────────────────────────────────────────────────

/// Compute spectral contrast.
///
/// For each sub-band, the contrast = mean of top-q peaks minus mean of bottom-q valleys.
/// Returns flat Vec<f32>, shape (n_bands+1, n_frames), row-major.
#[pyfunction]
#[pyo3(signature = (stft_mag, sr, n_fft, n_frames, n_bins, n_bands=6, fmin=200.0, quantile=0.02, linear=false))]
#[allow(clippy::too_many_arguments)]
pub fn spectral_contrast(
    stft_mag: Vec<f32>,
    sr: f32,
    n_fft: usize,
    n_frames: usize,
    n_bins: usize,
    n_bands: usize,
    fmin: f32,
    quantile: f32,
    linear: bool,
) -> Vec<f32> {
    let n_rows = n_bands + 1;
    let mut out = vec![0.0f32; n_rows * n_frames];

    // Frequency of each FFT bin
    let fft_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();

    // Sub-band edges (octave-spaced starting from fmin)
    let mut band_edges: Vec<f32> = (0..=n_bands)
        .map(|i| fmin * 2.0f32.powi(i as i32))
        .collect();
    // Last edge = Nyquist
    band_edges.push(sr / 2.0);

    for t in 0..n_frames {
        let frame = &stft_mag[t * n_bins..(t + 1) * n_bins];
        for band in 0..=n_bands {
            let flo = if band == 0 { 0.0 } else { band_edges[band - 1] };
            let fhi = band_edges[band].min(sr / 2.0);

            // Collect bin magnitudes (or energies) in this band
            let band_mags: Vec<f32> = fft_freqs
                .iter()
                .enumerate()
                .filter(|(_, &f)| f >= flo && f < fhi)
                .map(|(k, _)| {
                    if linear {
                        frame[k]
                    } else {
                        (frame[k] as f64 + 1e-10).ln() as f32
                    }
                })
                .collect();

            if band_mags.is_empty() {
                out[band * n_frames + t] = 0.0;
                continue;
            }

            let n = band_mags.len();
            let n_q = ((n as f32 * quantile).ceil() as usize).max(1).min(n);

            let mut sorted = band_mags.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let valley: f32 = sorted[..n_q].iter().sum::<f32>() / n_q as f32;
            let peak: f32 = sorted[n - n_q..].iter().sum::<f32>() / n_q as f32;

            out[band * n_frames + t] = peak - valley;
        }
    }

    out
}

// ─── Polynomial features ──────────────────────────────────────────────────────

/// Fit a polynomial of `order` to each spectral column.
/// Returns flat Vec<f32>, shape (order+1, n_frames) row-major.
/// Coefficients are from highest to lowest degree (like numpy.polyfit).
#[pyfunction]
#[pyo3(signature = (stft_mag, sr, n_fft, n_frames, n_bins, order=1))]
pub fn poly_features(
    stft_mag: Vec<f32>,
    sr: f32,
    n_fft: usize,
    n_frames: usize,
    n_bins: usize,
    order: usize,
) -> Vec<f32> {
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sr as f64 / n_fft as f64)
        .collect();
    let n_coeffs = order + 1;
    let mut out = vec![0.0f32; n_coeffs * n_frames];

    // Precompute Vandermonde-like system via least squares (normal equations)
    // For each frame, solve: V * c = y  where V[i,j] = freq[i]^j
    // Build VtV and Vty, solve by Gaussian elimination
    let n = n_bins;
    let m = n_coeffs;

    // Precompute powers of frequencies
    let mut freq_pows: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
    for (i, f) in fft_freqs.iter().copied().enumerate().take(n) {
        let mut p = 1.0f64;
        for row in freq_pows.iter_mut().take(m) {
            row[i] = p;
            p *= f;
        }
    }

    // VtV is (m × m), symmetric
    let mut vtv = vec![0.0f64; m * m];
    for a in 0..m {
        for b in 0..=a {
            let s: f64 = (0..n).map(|i| freq_pows[a][i] * freq_pows[b][i]).sum();
            vtv[a * m + b] = s;
            vtv[b * m + a] = s;
        }
    }

    for t in 0..n_frames {
        let frame = &stft_mag[t * n_bins..(t + 1) * n_bins];

        // Vty is (m,)
        let mut vty = vec![0.0f64; m];
        for a in 0..m {
            vty[a] = (0..n).map(|i| freq_pows[a][i] * frame[i] as f64).sum();
        }

        // Solve (VtV) * c = vty via Gaussian elimination with partial pivoting
        let coeffs = solve_linear(vtv.clone(), vty, m);

        // Coefficients from highest degree to lowest (like numpy.polyfit)
        for j in 0..m {
            out[(m - 1 - j) * n_frames + t] = coeffs[j] as f32;
        }
    }

    out
}

/// Simple Gaussian elimination with partial pivoting for small systems
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    // Augmented matrix: [a | b]
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in col + 1..n {
            if a[row * n + col].abs() > max_val {
                max_val = a[row * n + col].abs();
                max_row = row;
            }
        }
        // Swap rows
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in col + 1..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                let v = a[col * n + k] * factor;
                a[row * n + k] -= v;
            }
            b[row] -= b[col] * factor;
        }
    }
    // Back substitution
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        if a[row * n + row].abs() < 1e-12 {
            x[row] = 0.0;
            continue;
        }
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row * n + k] * x[k];
        }
        x[row] = s / a[row * n + row];
    }
    x
}

// ─── Delta features ───────────────────────────────────────────────────────────

/// Compute local estimate of the derivative of data along axis=-1.
///
/// Uses a linear regression approach over a window of `width` frames.
/// Parameters
/// ----------
/// data_flat  : flat Vec<f32>, shape (n_features, n_frames) row-major
/// n_features : number of feature rows
/// n_frames   : number of frames (time axis)
/// width      : number of frames used for local regression (must be odd, ≥3)
/// order      : order of difference (1=velocity, 2=acceleration)
///
/// Returns flat Vec<f32>, same shape as data.
#[pyfunction]
#[pyo3(signature = (data_flat, n_features, n_frames, width=9, order=1))]
pub fn delta(
    data_flat: Vec<f32>,
    n_features: usize,
    n_frames: usize,
    width: usize,
    order: usize,
) -> Vec<f32> {
    // Minimum width = 3, must be odd
    let w = width.max(3);
    let half = (w / 2) as isize;

    // Denominator: sum of squared offsets
    let denom: f32 = (1..=half as usize).map(|t| (t * t) as f32).sum::<f32>() * 2.0;
    let denom = if denom < 1e-8 { 1.0 } else { denom };

    let compute_single_delta = |input: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; n_features * n_frames];
        for feat in 0..n_features {
            for t in 0..n_frames as isize {
                let mut acc = 0.0f32;
                for offset in -half..=half {
                    if offset == 0 {
                        continue;
                    }
                    let idx = (t + offset).max(0).min(n_frames as isize - 1) as usize;
                    acc += offset as f32 * input[feat * n_frames + idx];
                }
                out[feat * n_frames + t as usize] = acc / denom;
            }
        }
        out
    };

    let first = compute_single_delta(&data_flat);
    if order <= 1 {
        return first;
    }
    // Recursively compute higher-order deltas
    compute_single_delta(&first)
}

// ─── Tempogram ────────────────────────────────────────────────────────────────

/// Compute an autocorrelation tempogram from an onset strength envelope.
///
/// Parameters
/// ----------
/// onset_env  : onset strength envelope, Vec<f32> of length n_onset_frames
/// sr         : sampling rate
/// hop_length : hop length used to compute onset_env
/// win_length : number of frames per tempogram frame (default 384)
/// center     : whether to pad the onset_env at both ends
///
/// Returns (flat Vec<f32>, n_tempo_bins, n_tg_frames)
/// The output shape is (win_length, n_tg_frames).
#[pyfunction]
#[pyo3(signature = (onset_env, sr, hop_length=512, win_length=384, center=true))]
pub fn tempogram(
    onset_env: Vec<f32>,
    sr: f32,
    hop_length: usize,
    win_length: usize,
    center: bool,
) -> (Vec<f32>, usize, usize) {
    let _ = sr;
    let n_onset = onset_env.len();
    if n_onset == 0 || win_length == 0 {
        return (vec![], 0, 0);
    }

    // Window function for tempogram frames
    let window = hann_window(win_length);
    let _ = hop_length;

    // Pad the onset envelope
    let padded: Vec<f32> = if center {
        let pad = win_length / 2;
        let mut p = vec![0.0f32; pad + n_onset + pad];
        p[pad..pad + n_onset].copy_from_slice(&onset_env);
        p
    } else {
        onset_env.clone()
    };

    let n_padded = padded.len();
    let tg_hop = 1;
    let n_tg_frames = if n_padded >= win_length {
        (n_padded - win_length) / tg_hop + 1
    } else {
        0
    };

    if n_tg_frames == 0 {
        return (vec![], win_length, 0);
    }

    // For each tempogram frame, compute autocorrelation of the windowed onset envelope
    let mut out = vec![0.0f32; win_length * n_tg_frames];

    for t in 0..n_tg_frames {
        let start = t * tg_hop;
        let frame: Vec<f32> = (0..win_length)
            .map(|i| padded[start + i] * window[i])
            .collect();

        // Autocorrelation via brute force for each lag
        let mean: f32 = frame.iter().sum::<f32>() / win_length as f32;
        let frame_centered: Vec<f32> = frame.iter().map(|&v| v - mean).collect();
        let var: f32 = frame_centered.iter().map(|&v| v * v).sum::<f32>();

        for lag in 0..win_length {
            let mut corr = 0.0f32;
            for i in 0..(win_length - lag) {
                corr += frame_centered[i] * frame_centered[i + lag];
            }
            out[lag * n_tg_frames + t] = if var > 1e-8 { corr / var } else { 0.0 };
        }
    }

    (out, win_length, n_tg_frames)
}

// ─── Beat tracking ────────────────────────────────────────────────────────────

/// Estimate the global tempo from an onset strength envelope.
///
/// Uses an autocorrelation-based approach in the lag domain, then finds the
/// best consistent lag near `start_bpm`.
///
/// Returns estimated BPM as f32.
#[pyfunction]
#[pyo3(signature = (onset_env, sr, hop_length=512, start_bpm=120.0, max_tempo=320.0))]
pub fn beat_tempo(
    onset_env: Vec<f32>,
    sr: f32,
    hop_length: usize,
    start_bpm: f32,
    max_tempo: f32,
) -> f32 {
    if onset_env.is_empty() {
        return start_bpm;
    }

    let n = onset_env.len();
    let frame_rate = sr / hop_length as f32; // frames per second

    // Lag range in frames corresponding to BPM range [10, max_tempo]
    let lag_min = (60.0 * frame_rate / max_tempo).floor() as usize;
    let lag_max = (60.0 * frame_rate / 10.0_f32).ceil() as usize;
    let lag_max = lag_max.min(n - 1);

    if lag_min >= lag_max {
        return start_bpm;
    }

    // Compute autocorrelation of onset envelope via FFT
    let fft_size = next_power_of_two(2 * n - 1);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut buf: Vec<Complex<f32>> = onset_env.iter().map(|&x| Complex::new(x, 0.0)).collect();
    buf.resize(fft_size, Complex::new(0.0, 0.0));
    fft.process(&mut buf);
    buf.iter_mut().for_each(|c| {
        let p = c.re * c.re + c.im * c.im;
        *c = Complex::new(p, 0.0);
    });
    ifft.process(&mut buf);
    let scale = fft_size as f32;
    let ac: Vec<f32> = buf[..n].iter().map(|c| c.re / scale).collect();

    // Gaussian prior centred at start_bpm
    let start_lag = 60.0 * frame_rate / start_bpm;
    let std_bpm = 0.3 * start_bpm;
    let std_lag = 60.0 * frame_rate / start_bpm.powi(2) * std_bpm * start_bpm;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_lag = lag_min;

    for lag in lag_min..=lag_max {
        let ac_val = if lag < ac.len() { ac[lag] } else { 0.0 };
        let prior = -0.5 * ((lag as f32 - start_lag) / std_lag.max(1.0)).powi(2);
        let score = ac_val * prior.exp();
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    if best_lag == 0 {
        return start_bpm;
    }

    60.0 * frame_rate / best_lag as f32
}

/// Beat tracking using dynamic programming on the onset strength envelope.
///
/// Returns Vec<u32> of beat frame indices.
#[pyfunction]
#[pyo3(signature = (onset_env, tempo, sr, hop_length=512, tightness=100.0, trim=true))]
pub fn beat_track_dp(
    onset_env: Vec<f32>,
    tempo: f32,
    sr: f32,
    hop_length: usize,
    tightness: f32,
    trim: bool,
) -> Vec<u32> {
    if onset_env.is_empty() || tempo <= 0.0 {
        return vec![];
    }

    let n = onset_env.len();
    let frame_rate = sr / hop_length as f32;
    let beat_period = (60.0 * frame_rate / tempo).round() as usize;

    if beat_period == 0 {
        return vec![];
    }

    // Normalize onset envelope to [0, 1]
    let max_val = onset_env.iter().cloned().fold(0.0f32, f32::max);
    let odf: Vec<f32> = if max_val > 1e-8 {
        onset_env.iter().map(|&v| v / max_val).collect()
    } else {
        onset_env.clone()
    };

    // Dynamic programming score and back-pointer
    let mut score = vec![f32::NEG_INFINITY; n];
    let mut backlink = vec![0i32; n];

    score[0] = odf[0];
    backlink[0] = -1;

    for t in 1..n {
        // Look back in range [beat_period/2, 2*beat_period]
        let lo = if t > beat_period * 2 {
            t - beat_period * 2
        } else {
            0
        };
        let hi = if t > beat_period / 2 {
            t - beat_period / 2
        } else {
            0
        };

        let mut best_score = f32::NEG_INFINITY;
        let mut best_prev = lo as i32;

        for (prev, prev_score) in score.iter().enumerate().take(hi + 1).skip(lo) {
            if *prev_score == f32::NEG_INFINITY {
                continue;
            }
            let delta = t as f32 - prev as f32;
            let penalty = -tightness * (((delta / beat_period as f32).ln()).powi(2));
            let s = *prev_score + penalty;
            if s > best_score {
                best_score = s;
                best_prev = prev as i32;
            }
        }

        score[t] = best_score + odf[t];
        backlink[t] = best_prev;
    }

    // Find the best final beat
    let mut best_t = 0;
    let mut best_s = f32::NEG_INFINITY;
    for (t, s) in score.iter().copied().enumerate().take(n) {
        if s > best_s {
            best_s = s;
            best_t = t;
        }
    }

    // Trace back to find all beat frames
    let mut beats = Vec::new();
    let mut t = best_t as i32;
    while t >= 0 {
        beats.push(t as u32);
        let prev = backlink[t as usize];
        if prev == t || prev < 0 {
            break;
        }
        t = prev;
    }
    beats.reverse();

    // Optionally trim leading/trailing beats below median onset strength
    if trim && !beats.is_empty() {
        let smooth_len = beat_period / 2;
        let threshold = {
            let vals: Vec<f32> = beats
                .iter()
                .map(|&b| {
                    let b = b as usize;
                    let lo = b.saturating_sub(smooth_len);
                    let hi = (b + smooth_len).min(n - 1);
                    odf[lo..=hi].iter().cloned().fold(0.0f32, f32::max)
                })
                .collect();
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 4] // 25th percentile
        };

        let first_valid = beats
            .iter()
            .position(|&b| odf[b as usize] >= threshold)
            .unwrap_or(0);
        let last_valid = beats
            .iter()
            .rposition(|&b| odf[b as usize] >= threshold)
            .unwrap_or(beats.len() - 1);

        beats = beats[first_valid..=last_valid].to_vec();
    }

    beats
}

// ─── Onset detection ──────────────────────────────────────────────────────────

/// Detect onsets in an onset strength envelope using peak picking.
///
/// Parameters
/// ----------
/// onset_env  : onset strength envelope
/// sr         : sample rate
/// hop_length : hop length used to compute onset_env
/// delta      : minimum height above local mean to be considered an onset (default 0.07)
/// wait       : number of frames to wait after each onset (default 30)
///
/// Returns Vec<u32> of onset frame indices.
#[pyfunction]
#[pyo3(signature = (onset_env, sr, hop_length=512, delta=0.07, wait=30))]
pub fn onset_detect(
    onset_env: Vec<f32>,
    sr: f32,
    hop_length: usize,
    delta: f32,
    wait: usize,
) -> Vec<u32> {
    let _ = sr;
    let _ = hop_length;

    let n = onset_env.len();
    if n < 2 {
        return vec![];
    }

    // Normalize to [0, 1]
    let max_val = onset_env.iter().cloned().fold(0.0f32, f32::max);
    let odf: Vec<f32> = if max_val > 1e-8 {
        onset_env.iter().map(|&v| v / max_val).collect()
    } else {
        return vec![];
    };

    // Compute local mean over a window
    let mean_window = (wait * 2).min(n);
    let local_mean: Vec<f32> = (0..n)
        .map(|i| {
            let lo = if i > mean_window / 2 {
                i - mean_window / 2
            } else {
                0
            };
            let hi = (i + mean_window / 2).min(n - 1);
            odf[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32
        })
        .collect();

    let mut onsets = Vec::new();
    let mut last_onset: Option<usize> = None;

    for i in 1..(n - 1) {
        // Peak picking: local maximum above threshold
        if odf[i] > odf[i - 1] && odf[i] >= odf[i + 1] && odf[i] > local_mean[i] + delta {
            if let Some(last) = last_onset {
                if i - last >= wait {
                    onsets.push(i as u32);
                    last_onset = Some(i);
                }
            } else {
                onsets.push(i as u32);
                last_onset = Some(i);
            }
        }
    }

    onsets
}

// Suppress unused import warning: PI64 is available for future use
#[allow(dead_code)]
const _PI64_CHECK: f64 = PI64;
