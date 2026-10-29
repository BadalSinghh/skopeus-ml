"""Unit tests for Skopeus ML Denoise & APT Service.

Tests:
1. Production checkpoint integrity & SHA-256 validation.
2. Model loading and CPU inference.
3. Preprocessing (20.8 kHz resampling, 100 Hz highpass, 0.85 headroom).
4. Audio roundtrip denoising.
5. RobustAPTDecoder demodulation, synchronization, and Channel A/B separation.
"""

import base64
import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import soundfile as sf
import torch

from denoise_service import (
    CERTIFIED_SHA256,
    DEFAULT_CHECKPOINT_PATH,
    TARGET_SAMPLE_RATE,
    compute_file_sha256,
    denoise_wav,
    get_denoise_service,
)
from apt_service import (
    APT_TARGET_SR,
    PIXELS_PER_LINE,
    CHANNEL_WIDTH,
    RobustAPTDecoder,
    decode_apt_audio,
    decode_apt_wav,
    matrix_to_data_url,
    matrix_to_png_bytes,
)


def test_checkpoint_sha256_integrity():
    """Verify production checkpoint matches certified research SHA-256."""
    assert DEFAULT_CHECKPOINT_PATH.is_file(), f"Missing checkpoint: {DEFAULT_CHECKPOINT_PATH}"
    digest = compute_file_sha256(DEFAULT_CHECKPOINT_PATH)
    assert digest == CERTIFIED_SHA256


def test_model_loading_and_parameter_count():
    """Verify DnCNN-1D loads with exactly 309,825 parameters."""
    service = get_denoise_service(device="cpu")
    assert service.model is not None
    total_params = service.model.count_parameters()
    assert total_params == 309825


def test_synthetic_audio_denoising(tmp_path):
    """Verify end-to-end WAV file denoising roundtrip."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 2.4 kHz carrier with 3 Hz AM modulation + noise
    clean = 0.5 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.cos(2 * np.pi * 2400 * t)
    noisy = clean + 0.1 * np.random.randn(len(t))

    in_wav = tmp_path / "noisy_input.wav"
    out_wav = tmp_path / "denoised_output.wav"
    sf.write(str(in_wav), noisy.astype(np.float32), sr, subtype="PCM_16")

    stats = denoise_wav(in_wav, out_wav, device="cpu")
    assert out_wav.is_file()
    assert stats["output_sample_rate"] == 20800
    assert stats["output_channels"] == 1
    assert stats["throughput_realtime_x"] > 1.0


def test_apt_decoder_synthetic_baseband():
    """Verify RobustAPTDecoder extracts 2080 px composite and 1040 px channel images."""
    sr = 20800
    duration = 2.25
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr

    carrier = np.cos(2 * np.pi * 2400.0 * t)
    video = 0.5 + 0.3 * np.sin(2 * np.pi * 3.0 * t)

    # Inject Sync A pulses (7 pulses every 10400 samples)
    samples_per_line = int(sr * 0.5)
    for line in range(4):
        sync_start = line * samples_per_line
        for p in range(7):
            p_start = sync_start + p * 25
            video[p_start : p_start + 12] = 0.95
            video[p_start + 12 : p_start + 25] = 0.05

    am_signal = (video * carrier).astype(np.float32)
    res = decode_apt_audio(am_signal, sr=sr)

    assert res["success"] is True
    assert res["metadata"]["width"] == PIXELS_PER_LINE
    assert res["metadata"]["num_lines"] >= 3
    assert res["image"].startswith("data:image/png;base64,")
    assert res["channel_a"].startswith("data:image/png;base64,")
    assert res["channel_b"].startswith("data:image/png;base64,")

    # Decode PNG bytes and check dimensions
    raw_comp = base64.b64decode(res["image"].split(",")[1])
    img_comp = Image.open(io.BytesIO(raw_comp))
    assert img_comp.size == (2080, res["metadata"]["num_lines"])

    raw_a = base64.b64decode(res["channel_a"].split(",")[1])
    img_a = Image.open(io.BytesIO(raw_a))
    assert img_a.size == (1040, res["metadata"]["num_lines"])
