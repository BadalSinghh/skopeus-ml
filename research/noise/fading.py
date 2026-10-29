"""Slow amplitude fading simulator (Faraday rotation and ground antenna pattern nulls)."""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def add_amplitude_fading(
    signal: np.ndarray,
    sr: int,
    min_gain: float = 0.35,
    max_gain: float = 1.0,
    fading_freq_hz: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply slowly time-varying multiplicative gain: $y(t) = a(t) \cdot x(t)$.

    Args:
        signal: 1D numpy array.
        sr: Sample rate in Hz.
        min_gain: Minimum gain floor.
        max_gain: Maximum gain ceiling.
        fading_freq_hz: Characteristic fading rate in Hz.
        seed: Random seed.

    Returns:
        (faded_signal, metadata)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(signal)
    if n_samples == 0:
        return signal, {"noise_type": "fading"}

    t = np.arange(n_samples, dtype=np.float32) / sr
    phase = float(rng.uniform(0, 2 * np.pi))
    freq = float(fading_freq_hz * rng.uniform(0.7, 1.3))

    # Smooth sinusoidal gain modulation bounded between [min_gain, max_gain]
    gain_envelope = 0.5 * (max_gain - min_gain) * (1.0 + np.sin(2.0 * np.pi * freq * t + phase)) + min_gain
    faded_signal = (signal * gain_envelope).astype(np.float32)

    meta = {
        "noise_type": "fading",
        "min_gain": min_gain,
        "max_gain": max_gain,
        "fading_frequency_hz": round(freq, 2),
        "seed": seed,
    }

    return faded_signal, meta
