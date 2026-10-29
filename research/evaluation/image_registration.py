"""Synchronization-Aware 2D Image Registration for Downstream Satellite Imagery Evaluation (Phase 11H).

Mitigates artificial metric penalties from subcarrier matched-filter phase jitter and line timing offsets.
Uses Fourier 2D Phase Correlation to estimate rigid translational displacements (dy, dx),
aligning decoded test images with reference images before computing PSNR and SSIM.
"""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.ndimage as ndi
from scipy.fft import fft2, ifft2, fftshift

from src.evaluation.apt_decoder import compute_image_psnr, compute_image_ssim


def phase_correlation_2d(
    ref: np.ndarray,
    test: np.ndarray,
    max_shift_y: int = 5,
    max_shift_x: int = 25,
) -> Tuple[float, float, float]:
    """Compute 2D rigid translation shift (dy, dx) using Normalized Cross-Power Spectrum.

    Args:
        ref: Reference 2D image matrix [H, W].
        test: Target/test 2D image matrix [H, W].
        max_shift_y: Maximum allowable vertical shift in lines.
        max_shift_x: Maximum allowable horizontal shift in pixels.

    Returns:
        (shift_y, shift_x, peak_correlation)
    """
    if ref.shape != test.shape:
        min_h = min(ref.shape[0], test.shape[0])
        min_w = min(ref.shape[1], test.shape[1])
        ref = ref[:min_h, :min_w]
        test = test[:min_h, :min_w]

    h, w = ref.shape
    if h < 2 or w < 2:
        return 0.0, 0.0, 1.0

    # Subtract mean to remove zero-frequency DC spike
    r_z = ref - np.mean(ref)
    t_z = test - np.mean(test)

    # 2D Hanning window to prevent edge boundary spectral ringing
    win_y = np.hanning(h)[:, None]
    win_x = np.hanning(w)[None, :]
    win_2d = (win_y * win_x).astype(np.float32)

    F_r = fft2(r_z * win_2d)
    F_t = fft2(t_z * win_2d)

    cross_power = F_r * np.conj(F_t)
    denom = np.abs(cross_power) + 1e-12
    normalized_cp = cross_power / denom

    corr_map = np.real(ifft2(normalized_cp))
    corr_map = fftshift(corr_map)

    center_y, center_x = h // 2, w // 2

    # Restrict search to realistic satellite sync jitter bounds
    y_min = max(0, center_y - max_shift_y)
    y_max = min(h, center_y + max_shift_y + 1)
    x_min = max(0, center_x - max_shift_x)
    x_max = min(w, center_x + max_shift_x + 1)

    sub_map = corr_map[y_min:y_max, x_min:x_max]
    peak_local = np.unravel_index(np.argmax(sub_map), sub_map.shape)

    peak_y = y_min + peak_local[0]
    peak_x = x_min + peak_local[1]

    shift_y = float(center_y - peak_y)
    shift_x = float(center_x - peak_x)
    peak_val = float(corr_map[peak_y, peak_x])

    return shift_y, shift_x, peak_val


def register_image(
    ref_image: np.ndarray,
    test_image: np.ndarray,
    max_shift_y: int = 5,
    max_shift_x: int = 25,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register test_image to ref_image by correcting 2D translation offset.

    Args:
        ref_image: Ground truth or high-SQI reference image [H, W].
        test_image: Decoded test/denoised image [H, W].

    Returns:
        (registered_test_image, registration_metadata)
    """
    min_h = min(ref_image.shape[0], test_image.shape[0])
    min_w = min(ref_image.shape[1], test_image.shape[1])

    r_crop = ref_image[:min_h, :min_w].astype(np.float32)
    t_crop = test_image[:min_h, :min_w].astype(np.float32)

    shift_y, shift_x, peak_corr = phase_correlation_2d(
        r_crop, t_crop, max_shift_y=max_shift_y, max_shift_x=max_shift_x
    )

    # Apply inverse rigid shift to align test image with reference
    registered = ndi.shift(t_crop, shift=(-shift_y, -shift_x), mode="nearest", order=1)
    registered = np.clip(registered, 0.0, 1.0)

    meta = {
        "shift_y_lines": round(shift_y, 2),
        "shift_x_pixels": round(shift_x, 2),
        "phase_peak_correlation": round(peak_corr, 4),
    }
    return registered, meta


def evaluate_dual_registration_metrics(
    ref_image: np.ndarray,
    test_image: np.ndarray,
) -> Dict[str, Any]:
    """Calculate downstream image metrics both BEFORE and AFTER 2D registration."""
    min_h = min(ref_image.shape[0], test_image.shape[0])
    min_w = min(ref_image.shape[1], test_image.shape[1])

    r_crop = ref_image[:min_h, :min_w]
    t_crop = test_image[:min_h, :min_w]

    # Unregistered metrics
    psnr_unreg = compute_image_psnr(r_crop, t_crop)
    ssim_unreg = compute_image_ssim(r_crop, t_crop)
    mae_unreg = float(np.mean(np.abs(r_crop - t_crop)))

    # Registered metrics
    reg_img, reg_meta = register_image(r_crop, t_crop)
    psnr_reg = compute_image_psnr(r_crop, reg_img)
    ssim_reg = compute_image_ssim(r_crop, reg_img)
    mae_reg = float(np.mean(np.abs(r_crop - reg_img)))

    return {
        "psnr_unregistered": round(psnr_unreg, 2),
        "psnr_registered": round(psnr_reg, 2),
        "delta_psnr_registration_gain": round(psnr_reg - psnr_unreg, 2),
        "ssim_unregistered": round(ssim_unreg, 4),
        "ssim_registered": round(ssim_reg, 4),
        "delta_ssim_registration_gain": round(ssim_reg - ssim_unreg, 4),
        "mae_unregistered": round(mae_unreg, 4),
        "mae_registered": round(mae_reg, 4),
        "shift_y_lines": reg_meta["shift_y_lines"],
        "shift_x_pixels": reg_meta["shift_x_pixels"],
        "phase_peak_correlation": reg_meta["phase_peak_correlation"],
    }
