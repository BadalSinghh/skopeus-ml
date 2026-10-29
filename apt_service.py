"""NOAA POES APT (Automatic Picture Transmission) Baseband Image Decoder & Service.

Decodes real-valued 2400 Hz AM baseband satellite audio signals into 2D visual imagery
(Visible Channel A + Thermal Infrared Channel B) and exports base64 PNG DataURLs.
Preserves exact research parameters:
- sr = 20800 Hz
- subcarrier = 2400 Hz
- 4th-order Butterworth lowpass at 1800 Hz+
- Normalized cross-correlation coarse Sync A detection
- Line tracking with +/-50 sample fine sync search
- 2080 pixels/line (1040 px Channel A, 1040 px Channel B)
- 2nd/98th percentile contrast stretching
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
from PIL import Image
import soundfile as sf

logger = logging.getLogger("skopeus.apt_service")

APT_TARGET_SR = 20800
SAMPLES_PER_LINE = 10400
PIXELS_PER_LINE = 2080
CHANNEL_WIDTH = 1040


class RobustAPTDecoder:
    """Scientific NOAA APT baseband audio-to-image decoder with normalized sync."""

    def __init__(self, sr: int = APT_TARGET_SR):
        self.sr = sr
        self.carrier_freq = 2400.0
        self.line_rate = 2.0  # 2 lines per second
        self.pixels_per_line = PIXELS_PER_LINE
        self.samples_per_line = int(self.sr / self.line_rate)  # 10,400
        self.samples_per_pixel = self.samples_per_line / self.pixels_per_line  # 5.0

        # Reference Sync A pulse train: 7 pulses of 832 Hz square wave
        # Period = 4160 / 832 = 5 pixels = 25 audio samples at 20.8 kHz
        sync_words = [0, 0, 255, 255, 0] * 7 + [0] * 4  # 39 words
        self.sync_ref_pixels = np.array(sync_words, dtype=np.float32) / 255.0
        self.sync_ref_samples = np.repeat(self.sync_ref_pixels, int(self.samples_per_pixel))

        # 4th-order Butterworth lowpass filter to remove 2400 Hz subcarrier ripple
        nyquist = sr / 2.0
        cutoff = 1800.0 / nyquist
        self.lp_b, self.lp_a = signal.butter(4, cutoff, btype="low")

    def extract_am_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Extract continuous instantaneous AM envelope via Hilbert analytic transform."""
        analytic = signal.hilbert(audio)
        envelope = np.abs(analytic)
        smooth_env = signal.filtfilt(self.lp_b, self.lp_a, envelope)
        return np.maximum(0.0, smooth_env)

    def decode(
        self,
        audio: np.ndarray,
        sync_search: bool = True,
        max_lines: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode audio into a 2D matrix using normalized coarse sync and fine line tracking."""
        envelope = self.extract_am_envelope(audio)
        n_samples = len(envelope)
        total_nominal_lines = n_samples // self.samples_per_line

        if total_nominal_lines == 0:
            return np.zeros((1, self.pixels_per_line), dtype=np.float32), {
                "num_lines": 0,
                "pixels_per_line": self.pixels_per_line,
                "avg_sync_correlation": 0.0,
                "sync_lock_percentage": 0.0,
            }

        num_lines = total_nominal_lines if max_lines is None else min(max_lines, total_nominal_lines)
        sync_len = len(self.sync_ref_samples)
        current_offset = 0

        # Normalized cross-correlation for coarse Sync A search across first 2 lines
        if sync_search and n_samples >= self.samples_per_line * 2:
            search_window = envelope[: self.samples_per_line * 2]
            corr = signal.correlate(search_window, self.sync_ref_samples, mode="valid")

            # Compute moving local energy of search window: sum(x[t+k]^2)
            energy = signal.convolve(
                search_window**2,
                np.ones(sync_len, dtype=np.float32),
                mode="valid",
            )
            norm_x = np.sqrt(np.maximum(0.0, energy))
            norm_s = float(np.linalg.norm(self.sync_ref_samples))

            denominator = norm_x * norm_s + 1e-6
            norm_corr = corr / denominator

            if len(norm_corr) > 0:
                best_start = int(np.argmax(norm_corr))
                current_offset = best_start

        lines = []
        sync_correlations = []

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
                        best_local = int(np.argmax(corr))
                        line_start = start_search + best_local
                        max_corr = float(corr[best_local]) / (
                            np.linalg.norm(slice_data[:sync_len]) * np.linalg.norm(self.sync_ref_samples) + 1e-6
                        )
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
            line_pixels = signal.resample(line_raw, self.pixels_per_line)
            lines.append(line_pixels)

        if not lines:
            return np.zeros((1, self.pixels_per_line), dtype=np.float32), {
                "num_lines": 0,
                "pixels_per_line": self.pixels_per_line,
                "avg_sync_correlation": 0.0,
                "sync_lock_percentage": 0.0,
            }

        image = np.array(lines, dtype=np.float32)

        # Contrast stretch: normalize using 2nd and 98th percentiles
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        image = np.clip(image, 0.0, 1.0)

        # Sync telemetry calculation
        locked_lines = sum(1 for c in sync_correlations if c >= 0.4)
        sync_lock_pct = (locked_lines / len(sync_correlations) * 100.0) if sync_correlations else 0.0
        avg_corr = float(np.mean(sync_correlations)) if sync_correlations else 0.0

        metadata = {
            "num_lines": len(lines),
            "pixels_per_line": self.pixels_per_line,
            "avg_sync_correlation": round(avg_corr, 4),
            "sync_lock_percentage": round(sync_lock_pct, 2),
        }

        return image, metadata


def matrix_to_png_bytes(matrix: np.ndarray) -> bytes:
    """Convert a 2D float matrix [0.0, 1.0] to PNG image bytes (8-bit grayscale)."""
    clipped = np.clip(matrix, 0.0, 1.0)
    uint8_img = (clipped * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(uint8_img, mode="L")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def matrix_to_data_url(matrix: np.ndarray) -> str:
    """Convert a 2D float matrix [0.0, 1.0] to a base64 PNG data URL."""
    png_bytes = matrix_to_png_bytes(matrix)
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def decode_apt_audio(
    audio: np.ndarray,
    sr: int = APT_TARGET_SR,
    max_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode 20.8 kHz audio array into NOAA APT imagery and channel maps."""
    if sr != APT_TARGET_SR:
        raise ValueError(
            f"APTDecoder requires standardized {APT_TARGET_SR} Hz audio, got {sr} Hz. "
            "Process audio through denoise_service first."
        )

    if audio.ndim > 1:
        if audio.shape[1] == 1:
            audio_1d = audio.squeeze(1).astype(np.float32)
        else:
            raise ValueError(f"APTDecoder requires mono audio, got {audio.shape[1]} channels.")
    else:
        audio_1d = audio.astype(np.float32)

    t_start = time.perf_counter()
    decoder = RobustAPTDecoder(sr=sr)
    full_matrix, meta = decoder.decode(audio_1d, sync_search=True, max_lines=max_lines)
    t_latency_ms = (time.perf_counter() - t_start) * 1000.0

    num_lines = meta["num_lines"]
    if num_lines == 0:
        return {
            "success": False,
            "error": "Insufficient audio duration (< 0.5 s / 10,400 samples) to decode an APT line.",
            "image": None,
            "channel_a": None,
            "channel_b": None,
            "raw_matrix": full_matrix,
            "metadata": {
                "sample_rate": sr,
                "num_lines": 0,
                "width": PIXELS_PER_LINE,
                "height": 0,
                "sync_lock_percentage": 0.0,
                "avg_sync_correlation": 0.0,
                "decoder_latency_ms": round(t_latency_ms, 2),
            },
        }

    # Slice Channel A (0:1040) and Channel B (1040:2080)
    chan_a_matrix = full_matrix[:, :CHANNEL_WIDTH]
    chan_b_matrix = full_matrix[:, CHANNEL_WIDTH:]

    full_url = matrix_to_data_url(full_matrix)
    chan_a_url = matrix_to_data_url(chan_a_matrix)
    chan_b_url = matrix_to_data_url(chan_b_matrix)

    metadata = {
        "sample_rate": sr,
        "num_lines": num_lines,
        "width": PIXELS_PER_LINE,
        "height": num_lines,
        "channel_a_width": CHANNEL_WIDTH,
        "channel_b_width": CHANNEL_WIDTH,
        "sync_lock_percentage": meta["sync_lock_percentage"],
        "avg_sync_correlation": meta["avg_sync_correlation"],
        "decoder_latency_ms": round(t_latency_ms, 2),
    }

    return {
        "success": True,
        "image": full_url,
        "channel_a": chan_a_url,
        "channel_b": chan_b_url,
        "raw_matrix": full_matrix,
        "metadata": metadata,
    }


def decode_apt_wav(
    wav_path: Union[str, Path],
    max_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode a cleaned 20.8 kHz WAV file into NOAA APT satellite imagery."""
    p = Path(wav_path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    audio, sr = sf.read(str(p), dtype="float32")
    return decode_apt_audio(audio=audio, sr=sr, max_lines=max_lines)
