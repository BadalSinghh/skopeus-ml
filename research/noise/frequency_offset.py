"""Frequency offset simulator adapted for real-valued audio signals."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal as signal


def apply_frequency_offset(
    audio: np.ndarray,
    sr: int,
    offset_hz: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Shift the frequency of real-valued audio using Single-Sideband (SSB) Hilbert modulation.

    For a real-valued signal $x(t)$ with analytic representation $x_a(t) = x(t) + j\mathcal{H}\{x(t)\}$:
    $$x_{\Delta f}(t) = \text{Re}\{x_a(t) \cdot e^{j 2\pi \Delta f t}\} = x(t)\cos(2\pi \Delta f t) - \mathcal{H}\{x(t)\}\sin(2\pi \Delta f t)$$

    This shifts all spectral components (including the 2400 Hz APT AM subcarrier) by $\Delta f$
    without generating negative alias frequencies or requiring complex I/Q inputs.

    Args:
        audio: 1D real-valued numpy array.
        sr: Sample rate in Hz.
        offset_hz: Frequency shift in Hz (e.g. $\pm 5\text{ Hz}$ to $\pm 20\text{ Hz}$ Doppler error).

    Returns:
        (frequency_shifted_audio, metadata)
    """
    n_samples = len(audio)
    if n_samples == 0 or abs(offset_hz) < 1e-6:
        return audio, {"noise_type": "frequency_offset", "offset_hz": 0.0}

    # Analytic signal via Hilbert transform
    analytic = signal.hilbert(audio)
    t = np.arange(n_samples, dtype=np.float32) / sr
    phasor = np.exp(1j * 2.0 * np.pi * offset_hz * t)

    shifted = np.real(analytic * phasor).astype(np.float32)

    meta = {
        "noise_type": "frequency_offset",
        "offset_hz": offset_hz,
    }

    return shifted, meta
