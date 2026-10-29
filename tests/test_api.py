"""FastAPI endpoint verification suite for Skopeus ML.

Tests:
1. GET /api/health
2. GET /api/ping
3. POST /api/denoise/upload with WAV file
4. POST /api/denoise/upload?decode=true (dual audio + imagery)
5. POST /api/decode/apt
"""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_wav_44k(tmp_path):
    """Generate 1.0 second 44.1 kHz mono WAV."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal_arr = 0.5 * np.cos(2 * np.pi * 2400.0 * t).astype(np.float32)
    p = tmp_path / "sample_44k.wav"
    sf.write(str(p), signal_arr, sr, subtype="PCM_16")
    return p


@pytest.fixture
def sample_apt_20k(tmp_path):
    """Generate 2.25 seconds 20.8 kHz synthetic APT baseband audio."""
    sr = 20800
    duration = 2.25
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr

    carrier = np.cos(2 * np.pi * 2400.0 * t)
    video = 0.5 + 0.3 * np.sin(2 * np.pi * 3.0 * t)

    samples_per_line = int(sr * 0.5)
    for line in range(4):
        sync_start = line * samples_per_line
        for p in range(7):
            p_start = sync_start + p * 25
            video[p_start : p_start + 12] = 0.95
            video[p_start + 12 : p_start + 25] = 0.05

    am_signal = (video * carrier).astype(np.float32)
    p = tmp_path / "sample_apt_20k.wav"
    sf.write(str(p), am_signal, sr, subtype="PCM_16")
    return p


def test_api_health(client):
    """GET /api/health returns HTTP 200 with model information."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["parameters"] == 309825


def test_api_ping(client):
    """GET /api/ping returns HTTP 200 for Render keepalive."""
    response = client.get("/api/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data


def test_api_upload_denoise_binary(client, sample_wav_44k):
    """POST /api/denoise/upload returns binary 20.8 kHz WAV by default."""
    with open(sample_wav_44k, "rb") as f:
        response = client.post(
            "/api/denoise/upload",
            files={"file": ("noisy.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    assert "audio/wav" in response.headers.get("content-type", "")
    assert response.headers.get("X-Output-Sample-Rate") == "20800"
    assert response.headers.get("X-Output-Channels") == "1"


def test_api_upload_denoise_with_decode(client, sample_apt_20k):
    """POST /api/denoise/upload?decode=true returns JSON bundle with audio and imagery."""
    with open(sample_apt_20k, "rb") as f:
        response = client.post(
            "/api/denoise/upload?decode=true",
            files={"audio_file": ("satellite.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "audio_data_url" in data
    assert data["audio_data_url"].startswith("data:audio/wav;base64,")
    assert data["image"].startswith("data:image/png;base64,")
    assert data["channel_a"].startswith("data:image/png;base64,")
    assert data["channel_b"].startswith("data:image/png;base64,")
    assert data["metadata"]["width"] == 2080


def test_api_decode_apt_standalone(client, sample_apt_20k):
    """POST /api/decode/apt decodes cleaned 20.8 kHz WAV into imagery."""
    with open(sample_apt_20k, "rb") as f:
        response = client.post(
            "/api/decode/apt",
            files={"audio_file": ("clean_20k.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["metadata"]["width"] == 2080
    assert data["metadata"]["num_lines"] >= 3
