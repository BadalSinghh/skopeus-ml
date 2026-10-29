"""Denoising Service Layer for Skopeus ML.

Centralized service abstraction for DnCNN-1D neural audio denoising.
Both POST /api/denoise/upload and POST /api/denoise/capture route through this single module.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

from models.dncnn1d import DnCNN1DDenoiser

logger = logging.getLogger("skopeus.denoise_service")

CERTIFIED_SHA256 = "24ce759369ad110b23a524bedc2ad90eb608b497801a3f2300861c1fb106a7c2"
_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = _BASE_DIR / "models" / "dncnn1d_production.pt"

# Research DSP Standardization Constants
TARGET_SAMPLE_RATE = 20800  # Exactly 5.0 samples/pixel for NOAA APT
WINDOW_SIZE = 4096          # Analysis window size
HOP_LENGTH = 2048           # 50% overlap
HEADROOM_FACTOR = 0.85      # Standardized research headroom scaling


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 digest of a file in 64 KB blocks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DenoiseService:
    """Singleton-style service wrapping the certified DnCNN-1D model."""

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.model: Optional[DnCNN1DDenoiser] = None
        self._load_and_verify_model()

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str.lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str.lower().startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device(device_str)
        else:
            return torch.device("cpu")

    def _load_and_verify_model(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Production checkpoint not found at: {self.checkpoint_path}."
            )

        actual_sha = compute_file_sha256(self.checkpoint_path)
        if actual_sha != CERTIFIED_SHA256:
            raise ValueError(
                f"Security check failed! Checkpoint SHA-256 mismatch.\n"
                f"Expected: {CERTIFIED_SHA256}\n"
                f"Got:      {actual_sha}"
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        arch = checkpoint.get("architecture", {})
        model = DnCNN1DDenoiser(
            in_channels=arch.get("in_channels", 1),
            hidden_channels=arch.get("hidden_channels", 64),
            num_layers=arch.get("num_layers", 17),
            kernel_size=arch.get("kernel_size", 5),
            norm_type=arch.get("norm_type", "batch"),
            residual_learning=arch.get("residual_learning", True),
        )

        state_dict = checkpoint["model_state_dict"]
        clean_state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        logger.info(
            "DnCNN-1D production model loaded on %s (SHA-256 verified: %s...)",
            self.device,
            actual_sha[:12],
        )

    def preprocess_audio(self, raw_audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, float]:
        """Apply standardized research preprocessing:

        1. Convert stereo/multichannel to mono.
        2. Polyphase resample to 20,800 Hz.
        3. 2nd-order Butterworth high-pass at 100 Hz.
        4. Headroom normalization (scale peak to 0.85).
        """
        # 1. Mono conversion
        if raw_audio.ndim > 1:
            audio_mono = np.mean(raw_audio, axis=1)
        else:
            audio_mono = raw_audio.copy()

        # 2. Polyphase resampling to exactly 20,800 Hz
        if orig_sr != TARGET_SAMPLE_RATE:
            num_target_samples = int(round(len(audio_mono) * TARGET_SAMPLE_RATE / orig_sr))
            audio_resampled = signal.resample(audio_mono, num_target_samples).astype(np.float32)
        else:
            audio_resampled = audio_mono.astype(np.float32)

        # 3. 2nd-order Butterworth high-pass filter at 100 Hz
        nyq = TARGET_SAMPLE_RATE / 2.0
        cutoff = 100.0 / nyq
        b, a = signal.butter(2, cutoff, btype="highpass")
        audio_filtered = signal.filtfilt(b, a, audio_resampled).astype(np.float32)

        # 4. Headroom scaling: scale peak to 0.85
        max_val = float(np.max(np.abs(audio_filtered)))
        if max_val > 1e-7:
            scale_factor = HEADROOM_FACTOR / max_val
            audio_normalized = audio_filtered * scale_factor
        else:
            scale_factor = 1.0
            audio_normalized = audio_filtered

        return audio_normalized, scale_factor

    def denoise_array(
        self,
        audio_normalized: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Run overlap-add neural inference using DnCNN-1D model."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        orig_len = len(audio_normalized)
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32)

        # Chunk into 4096-sample windows with 2048-sample hop
        starts = list(range(0, orig_len, HOP_LENGTH))
        windows = []
        for s in starts:
            chunk = audio_normalized[s : s + WINDOW_SIZE]
            if len(chunk) < WINDOW_SIZE:
                chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)), mode="constant")
            windows.append(chunk)

        windows_tensor = torch.from_numpy(np.stack(windows)).unsqueeze(1).float()
        hann = np.hanning(WINDOW_SIZE).astype(np.float32)
        hann_tensor = torch.from_numpy(hann).view(1, 1, -1)

        denoised_windows = []
        with torch.no_grad():
            for i in range(0, len(windows_tensor), batch_size):
                b_batch = windows_tensor[i : i + batch_size].to(self.device)
                denoised_b = self.model(b_batch)
                denoised_b = (denoised_b * hann_tensor.to(self.device)).cpu()
                denoised_windows.append(denoised_b.squeeze(1).numpy())

        all_denoised = np.concatenate(denoised_windows, axis=0)

        # Reconstruct waveform via weighted overlap-add
        out_len = starts[-1] + WINDOW_SIZE
        reconstructed = np.zeros(out_len, dtype=np.float32)
        window_weights = np.zeros(out_len, dtype=np.float32)

        for idx, s in enumerate(starts):
            reconstructed[s : s + WINDOW_SIZE] += all_denoised[idx]
            window_weights[s : s + WINDOW_SIZE] += hann

        window_weights = np.maximum(window_weights, 1e-8)
        reconstructed /= window_weights

        # Trim tail padding back to original length
        return reconstructed[:orig_len]

    def postprocess_audio(self, denoised_raw: np.ndarray, scale_factor: float) -> np.ndarray:
        """Reverse headroom scaling and clamp to [-1.0, 1.0]."""
        if scale_factor > 1e-7:
            restored = denoised_raw / scale_factor
        else:
            restored = denoised_raw
        return np.clip(restored, -1.0, 1.0).astype(np.float32)

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """End-to-end processing of a WAV file."""
        t_start = time.perf_counter()
        in_p = Path(input_path)
        out_p = Path(output_path)

        data, orig_sr = sf.read(str(in_p), dtype="float32")
        input_channels = 1 if data.ndim == 1 else data.shape[1]
        input_duration = len(data) / orig_sr

        audio_norm, scale_factor = self.preprocess_audio(data, orig_sr)
        denoised_raw = self.denoise_array(audio_norm)
        cleaned_audio = self.postprocess_audio(denoised_raw, scale_factor)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), cleaned_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

        elapsed = time.perf_counter() - t_start
        throughput = input_duration / elapsed if elapsed > 0 else 0.0

        return {
            "success": True,
            "input_file": str(in_p.name),
            "output_file": str(out_p.name),
            "input_sample_rate": orig_sr,
            "input_channels": input_channels,
            "input_duration_sec": round(input_duration, 3),
            "output_sample_rate": TARGET_SAMPLE_RATE,
            "output_channels": 1,
            "output_duration_sec": round(len(cleaned_audio) / TARGET_SAMPLE_RATE, 3),
            "device": str(self.device).upper(),
            "processing_time_sec": round(elapsed, 3),
            "throughput_realtime_x": round(throughput, 1),
            "checkpoint_sha256": CERTIFIED_SHA256,
        }


# Global cached service instance
_SERVICE_INSTANCE: Optional[DenoiseService] = None


def get_denoise_service(device: str = "auto") -> DenoiseService:
    """Retrieve or initialize the singleton DenoiseService."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = DenoiseService(device=device)
    elif device != "auto" and str(_SERVICE_INSTANCE.device).lower() != device.lower():
        _SERVICE_INSTANCE = DenoiseService(device=device)
    return _SERVICE_INSTANCE


def denoise_wav(
    input_wav_path: Union[str, Path],
    output_wav_path: Union[str, Path],
    device: str = "auto",
) -> Dict[str, Any]:
    """Convenience function matching research pipeline signature."""
    service = get_denoise_service(device=device)
    return service.process_file(input_wav_path, output_wav_path)
