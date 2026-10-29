r"""Signal and audio metric computation utilities for satellite denoising research."""

from typing import Dict, Any, Union
import numpy as np
import scipy.signal as signal
import torch


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ensure array is 1D float32 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).squeeze()
    return x


def calculate_mse(reference: Union[np.ndarray, torch.Tensor], estimate: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Mean Squared Error."""
    ref = to_numpy(reference)
    est = to_numpy(estimate)
    return float(np.mean(np.square(ref - est)))


def calculate_mae(reference: Union[np.ndarray, torch.Tensor], estimate: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Mean Absolute Error."""
    ref = to_numpy(reference)
    est = to_numpy(estimate)
    return float(np.mean(np.abs(ref - est)))


def calculate_rmse(reference: Union[np.ndarray, torch.Tensor], estimate: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(calculate_mse(reference, estimate)))


def calculate_snr(reference: Union[np.ndarray, torch.Tensor], estimate: Union[np.ndarray, torch.Tensor]) -> float:
    r"""Compute Signal-to-Noise Ratio (SNR) in decibels relative to reference signal.

    $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{\sum x^2}{\sum (x - \hat{x})^2}\right)$$
    """
    ref = to_numpy(reference)
    est = to_numpy(estimate)

    sig_power = float(np.sum(np.square(ref)))
    noise_power = float(np.sum(np.square(ref - est)))

    if noise_power <= 1e-12:
        return 100.0  # Perfect reconstruction ceiling
    if sig_power <= 1e-12:
        return -100.0

    snr = 10.0 * np.log10(sig_power / noise_power)
    return float(snr)


def calculate_snr_improvement(
    reference: Union[np.ndarray, torch.Tensor],
    noisy: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
) -> float:
    r"""Compute SNR Improvement (Delta SNR): $\Delta\text{SNR} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$."""
    input_snr = calculate_snr(reference, noisy)
    output_snr = calculate_snr(reference, denoised)
    return float(output_snr - input_snr)


def calculate_correlation(reference: Union[np.ndarray, torch.Tensor], estimate: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Pearson correlation coefficient."""
    ref = to_numpy(reference)
    est = to_numpy(estimate)
    if len(ref) < 2:
        return 0.0

    ref_centered = ref - np.mean(ref)
    est_centered = est - np.mean(est)
    denom = np.sqrt(np.sum(ref_centered**2) * np.sum(est_centered**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(ref_centered * est_centered) / denom)


def calculate_si_sdr(reference: Union[np.ndarray, torch.Tensor], estimate: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    ref = to_numpy(reference)
    est = to_numpy(estimate)

    ref_zero_mean = ref - np.mean(ref)
    est_zero_mean = est - np.mean(est)

    dot = np.sum(ref_zero_mean * est_zero_mean)
    ref_energy = np.sum(ref_zero_mean**2) + 1e-12

    # Scale factor alpha
    alpha = dot / ref_energy
    s_target = alpha * ref_zero_mean
    e_noise = est_zero_mean - s_target

    target_energy = np.sum(s_target**2)
    noise_energy = np.sum(e_noise**2) + 1e-12

    if target_energy <= 1e-12:
        return -100.0

    si_sdr = 10.0 * np.log10(target_energy / noise_energy)
    return float(si_sdr)


def calculate_spectral_convergence(
    reference: Union[np.ndarray, torch.Tensor],
    estimate: Union[np.ndarray, torch.Tensor],
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    r"""Compute Spectral Convergence: $\frac{\| |X| - |\hat{X}| \|_F}{\| |X| \|_F}$."""
    ref = to_numpy(reference)
    est = to_numpy(estimate)

    _, _, S_ref = signal.stft(ref, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, S_est = signal.stft(est, nperseg=n_fft, noverlap=n_fft - hop_length)

    mag_ref = np.abs(S_ref)
    mag_est = np.abs(S_est)

    diff_norm = np.linalg.norm(mag_ref - mag_est, "fro")
    ref_norm = np.linalg.norm(mag_ref, "fro") + 1e-12

    return float(diff_norm / ref_norm)


def calculate_all_metrics(
    reference: Union[np.ndarray, torch.Tensor],
    noisy: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Compute all standard evaluation metrics in a single call."""
    in_snr = calculate_snr(reference, noisy)
    out_snr = calculate_snr(reference, denoised)
    return {
        "input_snr_db": round(in_snr, 3),
        "output_snr_db": round(out_snr, 3),
        "snr_improvement_db": round(out_snr - in_snr, 3),
        "mse": round(calculate_mse(reference, denoised), 6),
        "mae": round(calculate_mae(reference, denoised), 6),
        "rmse": round(calculate_rmse(reference, denoised), 6),
        "correlation": round(calculate_correlation(reference, denoised), 4),
        "si_sdr_db": round(calculate_si_sdr(reference, denoised), 3),
        "spectral_convergence": round(calculate_spectral_convergence(reference, denoised), 4),
    }
