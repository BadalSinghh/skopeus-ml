"""Impulsive noise simulator (Bernoulli-Gaussian process)."""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def add_impulsive_noise(
    signal: np.ndarray,
    probability: float = 0.005,
    amplitude_factor: float = 1.5,
    min_duration: int = 1,
    max_duration: int = 4,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate atmospheric lightning discharges and automotive ignition pulses.

    Args:
        signal: 1D input numpy array.
        probability: Probability of impulse arrival per sample.
        amplitude_factor: Multiplier relative to signal peak amplitude.
        min_duration: Minimum duration of impulse in samples.
        max_duration: Maximum duration of impulse in samples.
        seed: Random seed for reproducibility.

    Returns:
        (noisy_signal, noise_metadata)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(signal)
    if n_samples == 0:
        return signal, {"noise_type": "impulsive", "num_impulses": 0}

    peak = float(np.max(np.abs(signal)))
    if peak < 1e-6:
        peak = 1.0

    noise = np.zeros(n_samples, dtype=np.float32)
    # Poisson / Bernoulli arrival
    arrival_mask = rng.random(n_samples) < probability
    arrival_indices = np.where(arrival_mask)[0]

    for idx in arrival_indices:
        dur = rng.integers(min_duration, max_duration + 1)
        sign = rng.choice([-1.0, 1.0])
        amp = sign * peak * amplitude_factor * rng.uniform(0.7, 1.3)
        end_idx = min(idx + dur, n_samples)
        # Apply exponential decay pulse
        pulse = amp * np.exp(-np.linspace(0, 2.0, end_idx - idx))
        noise[idx:end_idx] += pulse

    noisy_signal = (signal + noise).astype(np.float32)

    meta = {
        "noise_type": "impulsive",
        "probability": probability,
        "amplitude_factor": amplitude_factor,
        "num_impulses": int(len(arrival_indices)),
        "seed": seed,
    }

    return noisy_signal, meta
