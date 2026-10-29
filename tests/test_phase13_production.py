"""Unit tests for Phase 13: Production Model Export & Standalone Inference."""

import hashlib
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from models.dncnn1d import DnCNN1DDenoiser
from scripts.inference_dncnn import load_production_model, denoise_audio_array, denoise_wav


def test_production_checkpoint_integrity():
    """Verify production checkpoint file exists, has correct keys, and matches SHA-256."""
    ckpt_path = Path("models/dncnn1d_production.pt")
    sha_path = Path("models/dncnn1d_production.sha256")

    assert ckpt_path.exists(), f"Missing production checkpoint: {ckpt_path}"
    assert sha_path.exists(), f"Missing SHA-256 file: {sha_path}"

    # Verify SHA-256 hash matches
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    computed_hash = hasher.hexdigest()

    with open(sha_path, "r", encoding="utf-8") as f:
        expected_hash = f.read().split()[0].strip()

    assert computed_hash == expected_hash, f"Hash mismatch: {computed_hash} != {expected_hash}"

    # Verify payload schema
    payload = torch.load(ckpt_path, map_location="cpu")
    assert "version" in payload
    assert "architecture" in payload
    assert "preprocessing" in payload
    assert "chunking" in payload
    assert "model_state_dict" in payload
    assert payload["architecture"]["num_parameters"] == 309825


def test_production_model_loading():
    """Verify that production checkpoint instantiates and loads cleanly."""
    model, device, meta = load_production_model("models/dncnn1d_production.pt", device="cpu")
    assert isinstance(model, DnCNN1DDenoiser)
    assert device.type == "cpu"
    assert model.count_parameters() == 309825
    assert not model.training  # Must be in eval mode


def test_inference_pipeline_synthetic_signal():
    """Verify end-to-end denoising on a synthetic 2.4 kHz noisy subcarrier."""
    model, device, _ = load_production_model("models/dncnn1d_production.pt", device="cpu")

    # Generate 1 second of 2.4 kHz tone + noise at 20.8 kHz
    sr = 20800
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean = 0.5 * np.sin(2 * np.pi * 2400.0 * t)
    noise = 0.2 * np.random.randn(sr).astype(np.float32)
    noisy = clean + noise

    cleaned, out_sr, stats = denoise_audio_array(
        audio=noisy,
        orig_sr=sr,
        model=model,
        device=device,
        target_sr=20800,
        batch_size=16,
    )

    assert out_sr == 20800
    assert len(cleaned) == len(noisy)
    assert cleaned.dtype == np.float32
    assert stats["throughput_realtime_x"] > 1.0  # Faster than real-time


def test_denoise_wav_file_roundtrip(tmp_path):
    """Verify denoise_wav CLI/Python API creates a valid output WAV file."""
    sr = 20800
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig = 0.4 * np.sin(2 * np.pi * 2400.0 * t)

    input_wav = tmp_path / "synthetic_noisy.wav"
    output_wav = tmp_path / "synthetic_cleaned.wav"
    sf.write(str(input_wav), sig, sr, subtype="PCM_16")

    stats = denoise_wav(
        input_path=input_wav,
        output_path=output_wav,
        checkpoint_path="models/dncnn1d_production.pt",
        device="cpu",
        verbose=False,
    )

    assert output_wav.exists()
    assert output_wav.stat().st_size > 0
    read_data, read_sr = sf.read(str(output_wav))
    assert read_sr == 20800
    assert len(read_data) == len(sig)
    assert stats["output_sample_rate"] == 20800
