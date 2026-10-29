"""Classical DSP baseline filters for satellite audio denoising benchmark."""

from typing import Tuple, Dict, Any, Union
import numpy as np
import scipy.signal as signal
import torch


class NoProcessingBaseline:
    """Identity baseline (unprocessed noisy input)."""

    def __call__(self, noisy: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return noisy


class BandpassFilterBaseline:
    """Conventional Butterworth bandpass filter centered on NOAA APT subcarrier (1.0 kHz - 3.8 kHz)."""

    def __init__(self, sr: int = 20800, lowcut_hz: float = 1000.0, highcut_hz: float = 3800.0, order: int = 4):
        self.sr = sr
        self.lowcut = lowcut_hz
        self.highcut = highcut_hz
        self.order = order

        nyquist = 0.5 * sr
        low = lowcut_hz / nyquist
        high = min(0.499, highcut_hz / nyquist)
        self.b, self.a = signal.butter(order, [low, high], btype="bandpass")

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            orig_shape = x.shape
            orig_device = x.device
            arr = x.detach().cpu().numpy().squeeze()
        else:
            arr = np.asarray(x).squeeze()

        # Handle batch or single sample
        if arr.ndim == 1:
            filtered = signal.filtfilt(self.b, self.a, arr).astype(np.float32)
        elif arr.ndim == 2:
            filtered = np.empty_like(arr, dtype=np.float32)
            for i in range(len(arr)):
                filtered[i] = signal.filtfilt(self.b, self.a, arr[i])
        else:
            filtered = arr

        if is_torch:
            return torch.from_numpy(filtered).reshape(orig_shape).to(orig_device)
        return filtered


class SpectralGatingBaseline:
    """Spectral noise gating baseline via Short-Time Fourier Transform (STFT) thresholding."""

    def __init__(self, sr: int = 20800, n_fft: int = 512, hop_length: int = 128, threshold_factor: float = 1.8):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.threshold_factor = threshold_factor

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            orig_shape = x.shape
            orig_device = x.device
            arr = x.detach().cpu().numpy().squeeze()
        else:
            arr = np.asarray(x).squeeze()

        def _denoise_1d(sig_1d: np.ndarray) -> np.ndarray:
            if len(sig_1d) < self.n_fft:
                return sig_1d
            f, t, Zxx = signal.stft(sig_1d, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)

            # Estimate noise floor from bottom 15% quantile
            noise_thresh = np.quantile(mag, 0.15, axis=1, keepdims=True) * self.threshold_factor
            mask = mag > noise_thresh
            mag_clean = np.where(mask, mag - 0.5 * noise_thresh, 0.0)

            Zxx_clean = mag_clean * np.exp(1j * phase)
            _, rec = signal.istft(Zxx_clean, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
            return rec[:len(sig_1d)].astype(np.float32)

        if arr.ndim == 1:
            filtered = _denoise_1d(arr)
        elif arr.ndim == 2:
            filtered = np.empty_like(arr, dtype=np.float32)
            for i in range(len(arr)):
                filtered[i] = _denoise_1d(arr[i])
        else:
            filtered = arr

        if is_torch:
            return torch.from_numpy(filtered).reshape(orig_shape).to(orig_device)
        return filtered
