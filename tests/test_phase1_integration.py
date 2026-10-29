"""Phase 1 Backend & Integration Verification Suite.

Validates:
1. Health endpoint specification
2. Checkpoint existence & SHA-256 verification
3. Model architecture and weight parameters
4. CPU inference execution
5. CUDA inference execution (skipped cleanly when CUDA unavailable)
6. WAV upload endpoint (multipart/form-data)
7. Stereo audio handling (downmixing to mono)
8. 48,000 Hz SDR audio handling (resampling to 20,800 Hz)
9. Output WAV format (16-bit PCM mono 20,800 Hz)
10. Malformed WAV rejection (HTTP 400)
11. Non-WAV file rejection (HTTP 400)
12. Live SDR capture endpoint (raw binary audio/wav payload)
13. Unified denoising service invocation across endpoints
14. Research regression test on real historical NOAA satellite audio fixture
"""

import hashlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from server.app import app
from server.denoise_service import (
    CERTIFIED_SHA256,
    DEFAULT_CHECKPOINT_PATH,
    DenoiseService,
    compute_file_sha256,
    denoise_wav,
    resolve_device,
    verify_checkpoint,
)
from src.models.dncnn1d import DnCNN1DDenoiser


@pytest.fixture
def client():
    """FastAPI test client instance."""
    return TestClient(app)


@pytest.fixture
def synthetic_wav_20k(tmp_path):
    """Generate a clean synthetic 20.8 kHz mono WAV file (0.5 sec)."""
    sr = 20800
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 2400.0 * t)  # 2.4 kHz subcarrier
    wav_p = tmp_path / "test_20k.wav"
    sf.write(str(wav_p), sig, sr, subtype="PCM_16")
    return wav_p, sig, sr


@pytest.fixture
def synthetic_wav_48k_stereo(tmp_path):
    """Generate a 48 kHz stereo WAV file matching live SDR receiver format."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    left = 0.4 * np.sin(2 * np.pi * 2400.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 2400.0 * t + 0.2)
    stereo = np.stack([left, right], axis=1)
    wav_p = tmp_path / "test_48k_stereo.wav"
    sf.write(str(wav_p), stereo, sr, subtype="PCM_16")
    return wav_p, stereo, sr


# ==============================================================================
# 1. Health Endpoint & Checkpoint Verification
# ==============================================================================

def test_health_endpoint(client):
    """GET /api/health returns valid JSON with all required keys and verified status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "device" in data
    assert "cuda_available" in data
    assert isinstance(data["cuda_available"], bool)
    assert "gpu_name" in data
    assert "pytorch_version" in data
    assert "checkpoint" in data
    assert "checkpoint_sha256" in data
    assert "model_verified" in data
    assert data["model_verified"] is True
    assert data["checkpoint_sha256"].lower() == CERTIFIED_SHA256.lower()


def test_checkpoint_existence_and_sha256():
    """Verify physical checkpoint exists and exactly matches certified digest."""
    assert DEFAULT_CHECKPOINT_PATH.is_file(), f"Missing production checkpoint: {DEFAULT_CHECKPOINT_PATH}"

    computed_sha = compute_file_sha256(DEFAULT_CHECKPOINT_PATH)
    assert computed_sha.lower() == CERTIFIED_SHA256.lower()

    is_verified, ret_sha, msg = verify_checkpoint(DEFAULT_CHECKPOINT_PATH)
    assert is_verified is True
    assert ret_sha.lower() == CERTIFIED_SHA256.lower()
    assert "verified" in msg.lower()


def test_checkpoint_tamper_rejection(tmp_path):
    """Verify that any unverified or tampered checkpoint is strictly rejected."""
    tampered_p = tmp_path / "fake_model.pt"
    tampered_p.write_bytes(b"INVALID_WEIGHT_DATA_12345")

    is_verified, ret_sha, msg = verify_checkpoint(tampered_p)
    assert is_verified is False
    assert ret_sha != CERTIFIED_SHA256
    assert "mismatch" in msg.lower()

    # Verify DenoiseService refuses to load unverified checkpoint
    svc = DenoiseService(checkpoint_path=tampered_p)
    with pytest.raises(RuntimeError, match="Cannot load unverified production model"):
        svc.load_model()


# ==============================================================================
# 2. Model Architecture & Device Handling
# ==============================================================================

def test_model_architecture_and_parameters():
    """Verify loaded model matches exact research DnCNN-1D specs."""
    svc = DenoiseService.get_instance()
    model = svc.load_model(device="cpu")

    assert isinstance(model, DnCNN1DDenoiser)
    assert model.count_parameters() == 309825
    assert not model.training  # Must be in eval mode
    num_convs = sum(1 for m in model.network if isinstance(m, torch.nn.Conv1d))
    assert num_convs == 17


def test_cpu_inference(synthetic_wav_20k, tmp_path):
    """Verify inference executes cleanly on CPU and outputs valid finite audio."""
    wav_p, _, _ = synthetic_wav_20k
    out_p = tmp_path / "out_cpu.wav"

    stats = denoise_wav(
        input_wav_path=wav_p,
        output_wav_path=out_p,
        device="cpu",
    )

    assert out_p.is_file()
    assert stats["device"] == "CPU"
    assert stats["output_sample_rate"] == 20800
    assert stats["output_channels"] == 1

    cleaned, sr = sf.read(str(out_p))
    assert sr == 20800
    assert cleaned.ndim == 1
    assert np.all(np.isfinite(cleaned))
    assert np.max(np.abs(cleaned)) <= 1.0


def test_cuda_inference(tmp_path):
    """Verify CUDA inference when GPU is available, cleanly skip otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available in the current environment.")

    # Generate 2.0s signal to properly test GPU throughput beyond one-time kernel startup
    sr = 20800
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 2400.0 * t)
    wav_p = tmp_path / "test_cuda_2s.wav"
    sf.write(str(wav_p), sig, sr, subtype="PCM_16")

    out_p = tmp_path / "out_cuda.wav"

    stats = denoise_wav(
        input_wav_path=wav_p,
        output_wav_path=out_p,
        device="cuda",
    )

    assert out_p.is_file()
    assert "CUDA" in stats["device"]
    assert stats["gpu_name"] != "CPU"
    assert stats["throughput_realtime_x"] > 1.0  # Faster than real-time


def test_device_auto_resolution():
    """Verify resolve_device behavior for 'auto', 'cpu', and 'cuda'."""
    dev, name = resolve_device("auto")
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"

    dev_cpu, name_cpu = resolve_device("cpu")
    assert dev_cpu.type == "cpu"
    assert name_cpu == "CPU"


# ==============================================================================
# 3. HTTP Endpoints: Upload, Capture & File Handling
# ==============================================================================

def test_wav_upload_endpoint(client, synthetic_wav_20k):
    """POST /api/denoise/upload processes a valid WAV and returns audio/wav binary."""
    wav_p, _, _ = synthetic_wav_20k

    with open(wav_p, "rb") as f:
        response = client.post(
            "/api/denoise/upload",
            files={"audio_file": ("test_20k.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    assert "audio/wav" in response.headers.get("content-type", "")

    # Validate custom response headers
    assert response.headers.get("X-Output-Sample-Rate") == "20800"
    assert response.headers.get("X-Output-Channels") == "1"
    assert "X-Processing-Time-Seconds" in response.headers
    assert response.headers.get("X-Checkpoint-Sha256") == CERTIFIED_SHA256

    # Verify returned bytes form valid 20.8 kHz WAV
    raw_audio = io.BytesIO(response.content)
    cleaned, sr = sf.read(raw_audio)
    assert sr == 20800
    assert cleaned.ndim == 1
    assert len(cleaned) > 0
    assert np.all(np.isfinite(cleaned))


def test_stereo_and_48khz_input_handling(client, synthetic_wav_48k_stereo):
    """Upload endpoint must convert 48 kHz stereo input to 20.8 kHz mono output."""
    wav_p, _, _ = synthetic_wav_48k_stereo

    with open(wav_p, "rb") as f:
        response = client.post(
            "/api/denoise/upload",
            files={"file": ("sdr_capture_48k.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Input-Sample-Rate") == "48000"
    assert response.headers.get("X-Input-Channels") == "2"
    assert response.headers.get("X-Output-Sample-Rate") == "20800"
    assert response.headers.get("X-Output-Channels") == "1"

    raw_audio = io.BytesIO(response.content)
    cleaned, sr = sf.read(raw_audio)
    assert sr == 20800
    assert cleaned.ndim == 1


def test_malformed_wav_rejection(client):
    """Upload endpoint must reject corrupted bytes with HTTP 400 Bad Request."""
    corrupted_data = b"RIFF" + b"\x00" * 20 + b"CORRUPTED_PAYLOAD_NOT_WAV"
    response = client.post(
        "/api/denoise/upload",
        files={"audio_file": ("corrupted.wav", corrupted_data, "audio/wav")},
    )
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower() or "malformed" in response.json()["detail"].lower()


def test_non_wav_file_rejection(client):
    """Upload endpoint must reject non-wav file extensions with HTTP 400."""
    response = client.post(
        "/api/denoise/upload",
        files={"audio_file": ("test_recording.mp3", b"ID3_FAKE_MP3_STREAM", "audio/mpeg")},
    )
    assert response.status_code == 400
    assert "only .wav files are accepted" in response.json()["detail"].lower()


def test_capture_endpoint_raw_binary(client, synthetic_wav_20k):
    """POST /api/denoise/capture supports raw binary audio/wav body."""
    wav_p, _, _ = synthetic_wav_20k
    wav_bytes = wav_p.read_bytes()

    response = client.post(
        "/api/denoise/capture",
        content=wav_bytes,
        headers={"Content-Type": "audio/wav"},
    )
    assert response.status_code == 200
    assert "audio/wav" in response.headers.get("content-type", "")
    assert response.headers.get("X-Output-Sample-Rate") == "20800"


def test_capture_endpoint_multipart(client, synthetic_wav_20k):
    """POST /api/denoise/capture also supports multipart form data."""
    wav_p, _, _ = synthetic_wav_20k

    with open(wav_p, "rb") as f:
        response = client.post(
            "/api/denoise/capture",
            files={"file": ("live_capture.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    assert response.headers.get("X-Output-Sample-Rate") == "20800"


def test_same_service_used_by_upload_and_capture(client, synthetic_wav_20k):
    """Verify that both upload and capture endpoints call the exact same denoise_wav operation."""
    wav_p, _, _ = synthetic_wav_20k
    wav_bytes = wav_p.read_bytes()

    with patch("server.app.denoise_wav", wraps=denoise_wav) as mock_denoise:
        # 1. Call upload
        client.post(
            "/api/denoise/upload",
            files={"file": ("up.wav", wav_bytes, "audio/wav")},
        )
        assert mock_denoise.call_count == 1

        # 2. Call capture
        client.post(
            "/api/denoise/capture",
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        assert mock_denoise.call_count == 2


# ==============================================================================
# 4. Research Regression Test (Real Satellite Recording Fixture)
# ==============================================================================

def test_research_regression_historical_noaa_pass(tmp_path):
    """Regression test using real historical NOAA APT satellite WAV recording.

    Verifies:
    - Model loads without error
    - Preprocessing succeeds on real RF audio
    - Output is generated
    - Output duration is preserved
    - Output sample rate is 20,800 Hz
    - Output is strictly finite (no NaN / Inf)
    - Amplitude remains bounded in [-1.0, 1.0]
    """
    raw_fixture = Path("data/raw/audio_137620000Hz_18-06-06_24-01-2025.wav")
    if not raw_fixture.is_file():
        pytest.skip(f"Research fixture file not found at: {raw_fixture}")

    # Read first 2.0 seconds of the historical satellite pass
    data, sr = sf.read(str(raw_fixture), frames=int(2.0 * 20800), dtype="float32")
    fixture_slice = tmp_path / "satellite_slice.wav"
    sf.write(str(fixture_slice), data, sr, subtype="PCM_16")

    output_slice = tmp_path / "satellite_cleaned.wav"

    stats = denoise_wav(
        input_wav_path=fixture_slice,
        output_wav_path=output_slice,
        device="cpu",
    )

    assert output_slice.is_file()
    cleaned_audio, out_sr = sf.read(str(output_slice), dtype="float32")

    # 1. Sample rate strictly 20,800 Hz
    assert out_sr == 20800
    assert stats["output_sample_rate"] == 20800

    # 2. Output duration preserved
    input_dur = len(data) / float(sr)
    output_dur = len(cleaned_audio) / float(out_sr)
    assert abs(output_dur - input_dur) < 0.25, f"Duration mismatch: {output_dur:.2f}s vs {input_dur:.2f}s"

    # 3. Finite, no NaN or Inf
    assert not np.isnan(cleaned_audio).any(), "Cleaned audio contains NaN values"
    assert not np.isinf(cleaned_audio).any(), "Cleaned audio contains Inf values"

    # 4. Clamped to [-1.0, 1.0]
    assert np.max(np.abs(cleaned_audio)) <= 1.0, "Cleaned audio exceeds [-1, 1] range"

    # 5. Non-trivial energy restored
    assert np.std(cleaned_audio) > 1e-4, "Cleaned audio is silent / degenerate"
