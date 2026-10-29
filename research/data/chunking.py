"""Window slicing and Overlap-Add (OLA) signal reconstruction utilities."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def get_window(window_type: str, window_size: int) -> np.ndarray:
    """Generate analysis/synthesis window function."""
    if window_type == "hann":
        return np.hanning(window_size).astype(np.float32)
    elif window_type == "hamming":
        return np.hamming(window_size).astype(np.float32)
    elif window_type in ("rectangular", "none"):
        return np.ones(window_size, dtype=np.float32)
    elif window_type == "triangular":
        return np.bartlett(window_size).astype(np.float32)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def slice_into_windows(
    signal: np.ndarray,
    window_size: int = 4096,
    hop_size: int = 2048,
    pad: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """Slice a continuous 1D signal into uniform (overlapping) windows.

    Args:
        signal: 1D numpy array.
        window_size: Length of each window in samples.
        hop_size: Stride between successive window starts.
        pad: If True, zero-pad signal end so last samples are not dropped.

    Returns:
        (windows_array of shape [N, window_size], list_of_segment_metadata)
    """
    if len(signal) == 0:
        return np.empty((0, window_size), dtype=np.float32), []

    num_samples = len(signal)

    if pad and num_samples >= window_size:
        remainder = (num_samples - window_size) % hop_size
        if remainder != 0:
            pad_len = hop_size - remainder
            signal = np.pad(signal, (0, pad_len), mode="constant")
            num_samples = len(signal)
    elif num_samples < window_size:
        if pad:
            signal = np.pad(signal, (0, window_size - num_samples), mode="constant")
            num_samples = len(signal)
        else:
            return np.empty((0, window_size), dtype=np.float32), []

    num_windows = 1 + (num_samples - window_size) // hop_size
    windows = np.empty((num_windows, window_size), dtype=np.float32)
    metadata = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        windows[i] = signal[start:end]
        metadata.append({
            "window_idx": i,
            "start_sample": start,
            "end_sample": end,
        })

    return windows, metadata


def reconstruct_from_windows(
    windows: np.ndarray,
    hop_size: int = 2048,
    window_type: str = "hann",
    target_length: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct continuous 1D signal from windowed frames using Overlap-Add (OLA).

    Uses weighted window synthesis and linear normalization to eliminate boundary artifacts.

    Args:
        windows: 2D numpy array of shape [num_windows, window_size].
        hop_size: Hop size used during slicing.
        window_type: Synthesis window function ('hann', 'hamming', 'rectangular').
        target_length: Desired output length to trim padding.

    Returns:
        1D reconstructed numpy array.
    """
    if len(windows) == 0:
        return np.empty(0, dtype=np.float32)

    num_windows, window_size = windows.shape
    total_length = (num_windows - 1) * hop_size + window_size

    win_func = get_window(window_type, window_size)

    output = np.zeros(total_length, dtype=np.float32)
    norm = np.zeros(total_length, dtype=np.float32)

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        output[start:end] += windows[i] * win_func
        norm[start:end] += win_func

    # Avoid division by zero
    nonzero = norm > 1e-8
    output[nonzero] /= norm[nonzero]

    if target_length is not None and target_length <= total_length:
        output = output[:target_length]

    return output
