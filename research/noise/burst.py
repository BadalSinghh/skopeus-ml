"""Burst interference simulator (terrestrial VHF emitters, pagers, digital harmonics)."""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def add_burst_interference(
    signal: np.ndarray,
    sr: int,
    burst_probability: float = 0.3,
    min_duration_ms: float = 10.0,
    max_duration_ms: float = 60.0,
    burst_snr_db: float = -3.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject localized short-duration burst interference.

    Args:
        signal: 1D input numpy array.
        sr: Sample rate in Hz.
        burst_probability: Probability of a burst occurring in the segment.
        min_duration_ms: Minimum burst duration in milliseconds.
        max_duration_ms: Maximum burst duration in milliseconds.
        burst_snr_db: Local SNR during the burst duration.
        seed: Random seed.

    Returns:
        (noisy_signal, metadata)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(signal)
    if n_samples == 0:
        return signal, {"noise_type": "burst", "burst_occurred": False}

    if rng.random() > burst_probability:
        return signal.copy(), {"noise_type": "burst", "burst_occurred": False, "seed": seed}

    min_samples = max(4, int((min_duration_ms / 1000.0) * sr))
    max_samples = max(min_samples + 1, int((max_duration_ms / 1000.0) * sr))
    burst_len = min(n_samples // 2, rng.integers(min_samples, max_samples + 1))

    start_idx = rng.integers(0, max(1, n_samples - burst_len))
    end_idx = start_idx + burst_len

    # Local signal power
    local_sig = signal[start_idx:end_idx]
    local_power = float(np.mean(np.square(local_sig))) + 1e-6
    burst_power = local_power / (10.0 ** (burst_snr_db / 10.0))
    burst_std = np.sqrt(burst_power)

    # High-frequency modulated burst interference
    burst_noise = rng.normal(0.0, burst_std, size=burst_len).astype(np.float32)
    # Taper burst edges with Tukey/Hann window to avoid sharp transient clicks
    taper = np.hanning(burst_len)
    burst_noise *= taper

    noisy_signal = signal.copy()
    noisy_signal[start_idx:end_idx] += burst_noise

    meta = {
        "noise_type": "burst",
        "burst_occurred": True,
        "start_sample": int(start_idx),
        "duration_samples": int(burst_len),
        "duration_ms": round((burst_len / sr) * 1000.0, 2),
        "burst_snr_db": burst_snr_db,
        "seed": seed,
    }

    return noisy_signal.astype(np.float32), meta
