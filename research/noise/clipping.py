"""Receiver ADC / amplifier saturation and clipping simulator."""

from typing import Tuple, Dict, Any
import numpy as np


def apply_clipping(
    signal: np.ndarray,
    threshold: float = 0.85,
    mode: str = "hard",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate receiver baseband amplifier or ADC overload saturation.

    Args:
        signal: 1D numpy array.
        threshold: Amplitude threshold where saturation starts (0 < threshold <= 1.0).
        mode: 'hard' (standard clamp) or 'soft' (hyperbolic tangent / tanh compression).

    Returns:
        (clipped_signal, metadata)
    """
    if len(signal) == 0:
        return signal, {"noise_type": "clipping", "clipped_ratio": 0.0}

    threshold = max(0.1, min(1.0, threshold))

    if mode == "hard":
        clipped = np.clip(signal, -threshold, threshold).astype(np.float32)
    elif mode == "soft":
        # Smooth saturation using tanh: y = threshold * tanh(x / threshold)
        clipped = (threshold * np.tanh(signal / threshold)).astype(np.float32)
    else:
        raise ValueError(f"Unknown clipping mode: {mode}")

    clipped_count = int(np.sum(np.abs(signal) >= threshold))
    clipped_ratio = float(clipped_count / len(signal))

    meta = {
        "noise_type": "clipping",
        "clipping_threshold": threshold,
        "clipping_mode": mode,
        "clipped_samples": clipped_count,
        "clipped_ratio": round(clipped_ratio, 4),
    }

    return clipped, meta
