use pyo3::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
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

        for k in 0..n_bins {
            out.push(buf[k].re);
            out.push(buf[k].im);
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
            let mirror = Complex::new(stft_re[base + n_bins - 1 - k], -stft_im[base + n_bins - 1 - k]);
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
pub fn amplitude_to_db(
    s: Vec<f32>,
    ref_val: f32,
    amin: f32,
    top_db: Option<f32>,
) -> Vec<f32> {
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
pub fn power_to_db(
    s: Vec<f32>,
    ref_val: f32,
    amin: f32,
    top_db: Option<f32>,
) -> Vec<f32> {
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
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    let mut zc = vec![false; n];
    for i in 1..n {
        let a = if y[i - 1].abs() <= threshold { 0.0 } else { y[i - 1] };
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
            let sum_sq: f32 = y[start..start + frame_length]
                .iter()
                .map(|&v| v * v)
                .sum();
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
    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|k| k as f32 * sr / n_fft as f32)
        .collect();

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
pub fn mfcc(
    log_mel: Vec<f32>,
    n_mels: usize,
    n_frames: usize,
    n_mfcc: usize,
) -> Vec<f32> {
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
        let chroma_idx = ((midi_float.round() as i64 % n_chroma as i64 + n_chroma as i64) as usize) % n_chroma;
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
pub fn spectral_flatness(
    stft_mag: Vec<f32>,
    n_frames: usize,
    n_bins: usize,
) -> Vec<f32> {
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
pub fn onset_strength(
    stft_mag: Vec<f32>,
    n_frames: usize,
    n_bins: usize,
) -> Vec<f32> {
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

// Suppress unused import warning: PI64 is available for future use
#[allow(dead_code)]
const _PI64_CHECK: f64 = PI64;
