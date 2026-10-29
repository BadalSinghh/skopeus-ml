"""Signal preprocessing module for satellite baseband recordings."""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal as signal


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio signal using polyphase FIR filtering with anti-aliasing.

    Args:
        audio: 1D or 2D numpy array.
        orig_sr: Input sample rate in Hz.
        target_sr: Target output sample rate in Hz.

    Returns:
        Resampled numpy array with same dtype (float32).
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = signal.resample_poly(audio, up, down, axis=0)
    return resampled.astype(np.float32)


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono via averaging."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=-1).astype(np.float32)


def remove_dc_offset(audio: np.ndarray, sr: int, cutoff_hz: float = 100.0) -> np.ndarray:
    """Remove DC bias using a high-pass Butterworth filter.

    Falls back to mean subtraction if signal is too short for filtering.
    """
    if len(audio) == 0:
        return audio

    if len(audio) < 128:
        return audio - np.mean(audio)

    # 2nd-order Butterworth high-pass filter
    nyquist = 0.5 * sr
    norm_cutoff = min(cutoff_hz / nyquist, 0.49)
    b, a = signal.butter(2, norm_cutoff, btype="highpass")
    filtered = signal.filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def scale_headroom(audio: np.ndarray, target_peak: float = 0.85) -> Tuple[np.ndarray, float]:
    """Scale audio signal amplitude to guarantee target headroom.

    Prevents digital clipping wrap-around when synthetic noise is added.

    Args:
        audio: Input numpy array.
        target_peak: Target peak amplitude in range (0, 1.0].

    Returns:
        (scaled_audio, scaling_factor)
    """
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak <= 1e-12:
        return audio, 1.0

    scale_factor = target_peak / peak
    scaled = audio * scale_factor
    return scaled.astype(np.float32), scale_factor


def compute_sqi(segment: np.ndarray, sr: int) -> float:
    """Compute Signal Quality Index (SQI) as in-band to out-of-band power ratio in dB.

    In-band (APT image subcarrier): 1000 Hz to 3800 Hz.
    Out-of-band (FM noise floor): 5000 Hz to min(Nyquist, 10000 Hz).
    """
    if len(segment) < 256:
        return 0.0

    nperseg = min(len(segment), 2048)
    freqs, psd = signal.welch(segment, fs=sr, nperseg=nperseg)

    in_band = (freqs >= 1000) & (freqs <= 3800)
    out_band = (freqs >= 5000) & (freqs <= min(sr // 2, 10000))

    in_p = float(np.sum(psd[in_band])) if np.any(in_band) else 0.0
    out_p = float(np.sum(psd[out_band])) if np.any(out_band) else 1e-12

    sqi_db = 10.0 * np.log10(max(in_p, 1e-12) / max(out_p, 1e-12))
    return round(float(sqi_db), 2)


def preprocess_signal(
    raw_audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 20800,
    to_mono: bool = True,
    remove_dc: bool = True,
    dc_cutoff_hz: float = 100.0,
    headroom_scale: float = 0.85,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Complete standardized non-destructive signal preprocessing pipeline.

    Args:
        raw_audio: Input numpy array.
        orig_sr: Original sample rate in Hz.
        target_sr: Standard target sample rate in Hz.
        to_mono: Convert to mono if multi-channel.
        remove_dc: Apply highpass filter.
        dc_cutoff_hz: Highpass cutoff frequency.
        headroom_scale: Scale peak to provide headroom for additive noise.

    Returns:
        (preprocessed_audio, preprocessing_metadata)
    """
    # 1. Mono conversion
    mono_audio = convert_to_mono(raw_audio) if to_mono else raw_audio

    # 2. Resampling
    resampled_audio = resample_audio(mono_audio, orig_sr, target_sr)

    # 3. DC Removal
    clean_dc = remove_dc_offset(resampled_audio, target_sr, dc_cutoff_hz) if remove_dc else resampled_audio

    # 4. Headroom Scaling
    scaled_audio, scale_factor = scale_headroom(clean_dc, headroom_scale)

    meta = {
        "orig_sample_rate": orig_sr,
        "target_sample_rate": target_sr,
        "scale_factor": scale_factor,
        "orig_length_samples": len(raw_audio),
        "preprocessed_length_samples": len(scaled_audio),
        "preprocessed_duration_sec": round(len(scaled_audio) / target_sr, 3),
        "peak_amplitude": float(np.max(np.abs(scaled_audio))) if len(scaled_audio) > 0 else 0.0,
    }

    return scaled_audio, meta
