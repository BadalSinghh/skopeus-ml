"""Narrowband interference simulator (sinusoidal tones and carrier spikes)."""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np


def add_narrowband_interference(
    signal: np.ndarray,
    sr: int,
    num_tones: int = 1,
    freq_range_hz: Tuple[float, float] = (800.0, 4000.0),
    amplitude_ratio: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject discrete sinusoidal interference tones: $n(t) = \sum A_i \sin(2\pi f_i t + \phi_i)$.

    Args:
        signal: 1D numpy array.
        sr: Sample rate in Hz.
        num_tones: Number of interfering tones.
        freq_range_hz: Frequency span (f_min, f_max) in Hz.
        amplitude_ratio: Ratio of tone amplitude relative to signal RMS.
        seed: Random seed.

    Returns:
        (noisy_signal, metadata)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(signal)
    if n_samples == 0:
        return signal, {"noise_type": "narrowband", "num_tones": 0}

    sig_rms = float(np.sqrt(np.mean(np.square(signal)))) + 1e-6
    t = np.arange(n_samples, dtype=np.float32) / sr

    interference = np.zeros(n_samples, dtype=np.float32)
    tones_info: List[Dict[str, float]] = []

    f_min, f_max = freq_range_hz
    for _ in range(num_tones):
        freq = float(rng.uniform(f_min, f_max))
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = float(sig_rms * amplitude_ratio * rng.uniform(0.8, 1.2))

        interference += amp * np.sin(2.0 * np.pi * freq * t + phase)
        tones_info.append({"frequency_hz": round(freq, 1), "amplitude": round(amp, 4), "phase_rad": round(phase, 3)})

    noisy_signal = (signal + interference).astype(np.float32)

    meta = {
        "noise_type": "narrowband",
        "num_tones": num_tones,
        "tones": tones_info,
        "seed": seed,
    }

    return noisy_signal, meta
