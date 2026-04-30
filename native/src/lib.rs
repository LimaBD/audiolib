use pyo3::prelude::*;

mod dsp;
mod convert;

// ─── Module entry point ───────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // DSP functions
    m.add_function(wrap_pyfunction!(dsp::stft, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::istft, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::amplitude_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::power_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::db_to_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::db_to_power, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::zero_crossings, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::autocorrelate, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::mu_compress, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::mu_expand, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::to_mono, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::resample, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::get_rms, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::mel_filterbank, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::melspectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::mfcc, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::chroma_stft, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::spectral_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::spectral_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::spectral_rolloff, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::spectral_flatness, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::onset_strength, m)?)?;
    m.add_function(wrap_pyfunction!(dsp::yin, m)?)?;
    // Convert functions
    m.add_function(wrap_pyfunction!(convert::hz_to_mel, m)?)?;
    m.add_function(wrap_pyfunction!(convert::mel_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(convert::hz_to_midi, m)?)?;
    m.add_function(wrap_pyfunction!(convert::midi_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(convert::frames_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(convert::frames_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(convert::samples_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(convert::samples_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(convert::time_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(convert::time_to_samples, m)?)?;
    Ok(())
}
