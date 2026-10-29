"""Additive White Gaussian Noise (AWGN) channel simulator."""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Add White Gaussian Noise at an exact target Signal-to-Noise Ratio (SNR).

    $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

    Args:
        signal: 1D input numpy array.
        snr_db: Target SNR in decibels.
        seed: Optional random seed for deterministic noise generation.

    Returns:
        (noisy_signal, noise_metadata)
    """
    rng = np.random.default_rng(seed)

    sig_power = float(np.mean(np.square(signal)))
    if sig_power <= 1e-12:
        sig_power = 1e-6

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(0.0, noise_std, size=signal.shape).astype(np.float32)
    noisy_signal = (signal + noise).astype(np.float32)

    actual_noise_power = float(np.mean(np.square(noise)))
    actual_snr = 10.0 * np.log10(sig_power / max(actual_noise_power, 1e-12))

    meta = {
        "noise_type": "awgn",
        "target_snr_db": snr_db,
        "actual_snr_db": round(float(actual_snr), 3),
        "noise_power": float(actual_noise_power),
        "signal_power": float(sig_power),
        "seed": seed,
    }

    return noisy_signal, meta
