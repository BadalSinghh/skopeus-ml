"""NOAA POES APT (Automatic Picture Transmission) Baseband Image Decoder & Visual Quality Evaluator.

Decodes real-valued 2400 Hz AM baseband satellite audio signals into 2D visual imagery
(Visible Channel A + Thermal Infrared Channel B) and computes quantitative image
reconstruction metrics: PSNR, SSIM, and line synchronization lock stability.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter


class APTDecoder:
    """Deterministic scientific NOAA APT baseband audio-to-image decoder."""

    def __init__(self, sr: int = 20800):
        """Initialize APT Decoder.

        Args:
            sr: Sample rate in Hz (default: 20800 Hz, corresponding to exactly 5 samples/pixel).
        """
        self.sr = sr
        self.carrier_freq = 2400.0
        self.line_rate = 2.0  # 2 lines per second
        self.pixels_per_line = 2080
        self.samples_per_line = int(self.sr / self.line_rate)  # 10,400 samples at 20.8 kHz
        self.samples_per_pixel = self.samples_per_line / self.pixels_per_line  # 5.0

        # Construct reference Sync A pulse train (7 pulses of 832 Hz square wave)
        # Period = 4160 / 832 = 5 pixels = 25 audio samples at 20.8 kHz
        sync_words = [0, 0, 255, 255, 0] * 7 + [0] * 4  # 39 words
        self.sync_ref_pixels = np.array(sync_words, dtype=np.float32) / 255.0
        # Resample sync reference to audio sample rate (39 pixels * 5 samples = 195 samples)
        self.sync_ref_samples = np.repeat(self.sync_ref_pixels, int(self.samples_per_pixel))

        # 4th-order Butterworth lowpass filter to remove 2400 Hz subcarrier ripple
        nyquist = sr / 2.0
        cutoff = 1800.0 / nyquist
        self.lp_b, self.lp_a = signal.butter(4, cutoff, btype="low")

    def extract_am_envelope(self, audio: np.ndarray) -> np.ndarray:
        r"""Extract continuous instantaneous AM envelope via Hilbert analytic transform.

        $$e(t) = |x(t) + j \mathcal{H}[x](t)|$$
        """
        # Analytic signal
        analytic = signal.hilbert(audio)
        envelope = np.abs(analytic)
        # Lowpass filter to eliminate subcarrier ripple while retaining video bandwidth (< 2 kHz)
        smooth_env = signal.filtfilt(self.lp_b, self.lp_a, envelope)
        return np.maximum(0.0, smooth_env)

    def decode(
        self,
        audio: np.ndarray,
        sync_search: bool = True,
        max_lines: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode satellite baseband audio into a 2D image matrix.

        Args:
            audio: 1D baseband audio signal.
            sync_search: If True, search for line sync pulses for exact horizontal alignment.
            max_lines: Maximum number of scan lines to decode.

        Returns:
            (image_matrix, metadata) where image_matrix has shape [N_lines, 2080], normalized [0, 1].
        """
        envelope = self.extract_am_envelope(audio)
        n_samples = len(envelope)
        total_nominal_lines = n_samples // self.samples_per_line

        if total_nominal_lines == 0:
            return np.zeros((1, self.pixels_per_line), dtype=np.float32), {"num_lines": 0}

        if max_lines is not None:
            num_lines = min(max_lines, total_nominal_lines)
        else:
            num_lines = total_nominal_lines

        lines = []
        sync_correlations = []
        current_offset = 0

        # Initial coarse Sync A search in first 2 lines
        if sync_search and n_samples >= self.samples_per_line * 2:
            search_window = envelope[: self.samples_per_line * 2]
            corr = signal.correlate(search_window, self.sync_ref_samples, mode="valid")
            if len(corr) > 0:
                best_start = int(np.argmax(corr))
                current_offset = best_start

        sync_len = len(self.sync_ref_samples)

        for line_idx in range(num_lines):
            nominal_start = current_offset + line_idx * self.samples_per_line

            # Fine sync search around expected line boundary (+/- 50 samples)
            if sync_search and nominal_start + self.samples_per_line + sync_len <= n_samples:
                search_radius = 50
                start_search = max(0, nominal_start - search_radius)
                end_search = min(n_samples, nominal_start + sync_len + search_radius)
                slice_data = envelope[start_search:end_search]

                if len(slice_data) >= sync_len:
                    corr = signal.correlate(slice_data, self.sync_ref_samples, mode="valid")
                    if len(corr) > 0:
                        best_local = np.argmax(corr)
                        line_start = start_search + best_local
                        max_corr = float(corr[best_local]) / (np.linalg.norm(slice_data[:sync_len]) * np.linalg.norm(self.sync_ref_samples) + 1e-6)
                        sync_correlations.append(max_corr)
                    else:
                        line_start = nominal_start
                        sync_correlations.append(0.0)
                else:
                    line_start = nominal_start
                    sync_correlations.append(0.0)
            else:
                line_start = nominal_start
                sync_correlations.append(1.0)

            line_end = line_start + self.samples_per_line
            if line_end > n_samples:
                break

            line_raw = envelope[line_start:line_end]
            # Resample line audio samples (10400) to exact standard APT pixel count (2080)
            line_pixels = signal.resample(line_raw, self.pixels_per_line)
            lines.append(line_pixels)

        if not lines:
            return np.zeros((1, self.pixels_per_line), dtype=np.float32), {"num_lines": 0}

        image = np.array(lines, dtype=np.float32)

        # Contrast stretch: normalize using 2nd and 98th percentiles
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        image = np.clip(image, 0.0, 1.0)

        meta = {
            "num_lines": len(lines),
            "pixels_per_line": self.pixels_per_line,
            "avg_sync_correlation": round(float(np.mean(sync_correlations)), 4) if sync_correlations else 0.0,
            "sync_lock_percentage": round(float(np.mean([c > 0.4 for c in sync_correlations])) * 100.0, 1) if sync_correlations else 0.0,
        }

        return image, meta


def compute_image_psnr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    r"""Compute 2D Peak Signal-to-Noise Ratio (PSNR in dB).

    $$\text{PSNR} = 10 \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$
    """
    assert ref_img.shape == test_img.shape, f"Shape mismatch: {ref_img.shape} vs {test_img.shape}"
    mse = float(np.mean((ref_img - test_img) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_image_ssim(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    sigma: float = 1.5,
    c1: float = (0.01) ** 2,
    c2: float = (0.03) ** 2,
) -> float:
    """Compute 2D Structural Similarity Index (SSIM) using Gaussian spatial weighting.

    Follows Wang et al., 'Image Quality Assessment: From Error Visibility to Structural Similarity', IEEE TIP 2004.
    """
    assert ref_img.shape == test_img.shape, f"Shape mismatch: {ref_img.shape} vs {test_img.shape}"

    # Local means
    mu1 = gaussian_filter(ref_img, sigma=sigma)
    mu2 = gaussian_filter(test_img, sigma=sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = gaussian_filter(ref_img * ref_img, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(test_img * test_img, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(ref_img * test_img, sigma=sigma) - mu1_mu2

    # SSIM map
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)

    return float(np.mean(ssim_map))
