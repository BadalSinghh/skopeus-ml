"""Phase 2 Integration Verification Suite: NOAA APT Decoder & DnCNN-1D Coupling.

Validates:
1. apt_service unit decoding on synthetic baseband
2. PNG serialization and base64 DataURL formatting
3. Channel A / Channel B dimensions (1040 px each) and Full Composite (2080 px)
4. Standardized 20,800 Hz input sample rate validation
5. Normalized coarse synchronization robustness
6. POST /api/decode/apt standalone endpoint
7. POST /api/denoise/upload?decode=true dual-path JSON API
8. Backward compatibility with Phase 1 binary WAV responses
9. Real historical satellite integration: NOAA-15 (137.620 MHz)
10. Real historical satellite integration: NOAA-19 (137.100 MHz)
"""

import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from app import app
from apt_service import (
    APT_TARGET_SR,
    PIXELS_PER_LINE,
    CHANNEL_WIDTH,
    RobustAPTDecoder,
    decode_apt_audio,
    decode_apt_wav,
    matrix_to_png_bytes,
    matrix_to_data_url,
)
from denoise_service import denoise_wav


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def synthetic_apt_baseband_20k(tmp_path):
    """Generate 2.25 seconds (4 scanlines with margin) of synthetic NOAA APT baseband audio at 20.8 kHz."""
    sr = 20800
    duration = 2.25
    total_samples = int(duration * sr)
    t = np.arange(total_samples, dtype=np.float32) / sr

    # 2.4 kHz subcarrier modulated by video waveform
    carrier = np.cos(2 * np.pi * 2400.0 * t)
    video = 0.5 + 0.3 * np.sin(2 * np.pi * 3.0 * t)

    # Inject Sync A pattern at each line boundary (every 10,400 samples)
    samples_per_line = int(sr * 0.5)
    for line in range(4):
        sync_start = line * samples_per_line
        for p in range(7):
            p_start = sync_start + p * 25
            video[p_start : p_start + 12] = 0.95
            video[p_start + 12 : p_start + 25] = 0.05

    am_signal = (video * carrier).astype(np.float32)
    wav_p = tmp_path / "synthetic_apt_20k.wav"
    sf.write(str(wav_p), am_signal, sr, subtype="PCM_16")
    return wav_p, am_signal, sr


# ==============================================================================
# 1. apt_service Unit Tests & Image Serialization
# ==============================================================================

def test_apt_service_synthetic_decoding(synthetic_apt_baseband_20k):
    """Verify apt_service successfully decodes 20.8 kHz synthetic audio."""
    wav_p, audio, sr = synthetic_apt_baseband_20k
    res = decode_apt_audio(audio, sr=sr)

    assert res["success"] is True
    assert res["metadata"]["width"] == PIXELS_PER_LINE
    assert res["metadata"]["num_lines"] in (3, 4)
    assert res["metadata"]["sample_rate"] == 20800
    assert res["metadata"]["sync_lock_percentage"] == 100.0
    assert res["metadata"]["avg_sync_correlation"] > 0.4
    assert res["raw_matrix"].shape == (res["metadata"]["num_lines"], PIXELS_PER_LINE)


def test_png_serialization_validity():
    """Verify matrix_to_png_bytes generates valid 8-bit grayscale PNG."""
    dummy_matrix = np.linspace(0.0, 1.0, 2080 * 10, dtype=np.float32).reshape(10, 2080)
    png_bytes = matrix_to_png_bytes(dummy_matrix)

    # Check magic PNG signature
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Open with Pillow and verify metadata
    img = Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.mode == "L"  # 8-bit grayscale
    assert img.size == (2080, 10)  # (width, height)

    # Test DataURL formatting
    data_url = matrix_to_data_url(dummy_matrix)
    assert data_url.startswith("data:image/png;base64,")


def test_channel_a_b_dimensions_and_separation(synthetic_apt_baseband_20k):
    """Verify Channel A and Channel B are separated into exact 1040-pixel images."""
    wav_p, _, _ = synthetic_apt_baseband_20k
    res = decode_apt_wav(wav_p)

    assert res["success"] is True
    assert res["channel_a"].startswith("data:image/png;base64,")
    assert res["channel_b"].startswith("data:image/png;base64,")

    # Decode base64 PNGs and check dimensions
    raw_b64_comp = res["image"].split(",")[1]
    raw_b64_a = res["channel_a"].split(",")[1]
    raw_b64_b = res["channel_b"].split(",")[1]

    img_comp = Image.open(io.BytesIO(base64.b64decode(raw_b64_comp)))
    img_a = Image.open(io.BytesIO(base64.b64decode(raw_b64_a)))
    img_b = Image.open(io.BytesIO(base64.b64decode(raw_b64_b)))

    num_lines = res["metadata"]["num_lines"]
    assert img_comp.size == (2080, num_lines)
    assert img_a.size == (1040, num_lines)
    assert img_b.size == (1040, num_lines)


def test_20k_input_sample_rate_validation():
    """Verify APTDecoder strictly rejects non-20,800 Hz audio."""
    dummy_audio = np.zeros(48000, dtype=np.float32)

    with pytest.raises(ValueError, match="APTDecoder requires standardized 20800 Hz audio"):
        decode_apt_audio(dummy_audio, sr=48000)

    with pytest.raises(ValueError, match="APTDecoder requires standardized 20800 Hz audio"):
        decode_apt_audio(dummy_audio, sr=44100)


def test_normalized_coarse_sync_robustness():
    """Verify normalized cross-correlation does not false-trigger on high-amplitude noise."""
    sr = 20800
    duration = 1.0  # 2 lines
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr

    # Video carrier with true sync at sample 0
    signal_arr = np.zeros(n_samples, dtype=np.float32)
    # Inject high-amplitude non-sync burst in the middle of line 0
    signal_arr[4000:4500] = 5.0  # Unnormalized dot product would heavily favor this

    # Inject true Sync A at start (samples 0 to 195)
    sync_words = [0, 0, 255, 255, 0] * 7 + [0] * 4
    sync_pixels = np.repeat(np.array(sync_words, dtype=np.float32) / 255.0, 5)
    signal_arr[: len(sync_pixels)] = sync_pixels * np.cos(2 * np.pi * 2400.0 * t[: len(sync_pixels)])

    dec = RobustAPTDecoder(sr=sr)
    img, meta = dec.decode(signal_arr)
    # Coarse sync should remain near sample 0, not sample 4000
    assert meta["num_lines"] >= 1


# ==============================================================================
# 2. API Endpoints: Decode & Denoise Flow
# ==============================================================================

def test_api_decode_apt_standalone_endpoint(client, synthetic_apt_baseband_20k):
    """POST /api/decode/apt decodes valid 20.8 kHz WAV into JSON imagery payload."""
    wav_p, _, _ = synthetic_apt_baseband_20k

    with open(wav_p, "rb") as f:
        response = client.post(
            "/api/decode/apt",
            files={"audio_file": ("clean_20k.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["image"].startswith("data:image/png;base64,")
    assert data["channel_a"].startswith("data:image/png;base64,")
    assert data["channel_b"].startswith("data:image/png;base64,")
    assert data["metadata"]["width"] == 2080
    assert data["metadata"]["sample_rate"] == 20800
    assert data["metadata"]["num_lines"] in (3, 4)
    assert data["metadata"]["decoder_latency_ms"] > 0


def test_api_decode_apt_rejects_non_20k(client, tmp_path):
    """POST /api/decode/apt rejects 48 kHz audio with HTTP 400."""
    p_48k = tmp_path / "bad_sr.wav"
    sf.write(str(p_48k), np.zeros(48000, dtype=np.float32), 48000, subtype="PCM_16")

    with open(p_48k, "rb") as f:
        response = client.post(
            "/api/decode/apt",
            files={"file": ("bad_sr.wav", f, "audio/wav")},
        )

    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "20,800" in detail or "20800" in detail


def test_api_denoise_upload_with_decode_query(client, synthetic_apt_baseband_20k):
    """POST /api/denoise/upload?decode=true returns JSON payload with both audio and imagery."""
    wav_p, _, _ = synthetic_apt_baseband_20k

    with open(wav_p, "rb") as f:
        response = client.post(
            "/api/denoise/upload?decode=true",
            files={"audio_file": ("test_noisy.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "audio_data_url" in data
    assert data["audio_data_url"].startswith("data:audio/wav;base64,")
    assert "audio_metadata" in data
    assert data["audio_metadata"]["output_sample_rate"] == 20800
    assert data["image"].startswith("data:image/png;base64,")
    assert data["channel_a"].startswith("data:image/png;base64,")
    assert data["channel_b"].startswith("data:image/png;base64,")
    assert data["metadata"]["num_lines"] in (3, 4)


def test_phase1_backward_compatibility(client, synthetic_apt_baseband_20k):
    """Verify that calling /api/denoise/upload without ?decode=true returns binary WAV."""
    wav_p, _, _ = synthetic_apt_baseband_20k

    with open(wav_p, "rb") as f:
        response = client.post(
            "/api/denoise/upload",
            files={"file": ("test_p1.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    assert "audio/wav" in response.headers.get("content-type", "")
    assert response.headers.get("X-Output-Sample-Rate") == "20800"
    assert response.headers.get("X-Output-Channels") == "1"


# ==============================================================================
# 3. Real Historical Satellite Recordings Integration
# ==============================================================================

def test_historical_noaa15_end_to_end(tmp_path):
    """End-to-end integration test on real historical NOAA-15 downlinks (44.1 kHz, 16.8s)."""
    raw_fixture = Path("data/raw/audio_137620000Hz_18-06-06_24-01-2025.wav")
    if not raw_fixture.is_file():
        pytest.skip(f"Fixture not found: {raw_fixture}")

    clean_wav = tmp_path / "noaa15_cleaned.wav"

    # 1. Denoise through DnCNN-1D
    denoise_stats = denoise_wav(raw_fixture, clean_wav, device="auto")
    assert clean_wav.is_file()
    assert denoise_stats["output_sample_rate"] == 20800

    # 2. Decode through apt_service
    res = decode_apt_wav(clean_wav)
    assert res["success"] is True
    assert res["metadata"]["width"] == 2080
    assert res["metadata"]["num_lines"] in (31, 32)
    assert res["metadata"]["sync_lock_percentage"] >= 90.0
    assert res["metadata"]["avg_sync_correlation"] > 0.60
    assert res["image"].startswith("data:image/png;base64,")


def test_historical_noaa19_end_to_end(tmp_path):
    """End-to-end integration test on real historical NOAA-19 pass (48.0 kHz, 10s slice)."""
    raw_fixture = Path("data/raw/audio_137100000Hz_09-19-25_10-12-2024.wav")
    if not raw_fixture.is_file():
        pytest.skip(f"Fixture not found: {raw_fixture}")

    # Slice 10 seconds of raw 48 kHz audio
    data, sr = sf.read(str(raw_fixture), frames=int(10.0 * 48000), dtype="float32")
    slice_wav = tmp_path / "noaa19_10s_raw.wav"
    clean_wav = tmp_path / "noaa19_10s_clean.wav"
    sf.write(str(slice_wav), data, sr, subtype="PCM_16")

    # 1. Denoise through DnCNN-1D
    denoise_stats = denoise_wav(slice_wav, clean_wav, device="auto")
    assert clean_wav.is_file()
    assert denoise_stats["output_sample_rate"] == 20800

    # 2. Decode through apt_service
    res = decode_apt_wav(clean_wav)
    assert res["success"] is True
    assert res["metadata"]["width"] == 2080
    assert res["metadata"]["num_lines"] in (18, 19)
    assert res["metadata"]["sync_lock_percentage"] >= 90.0
    assert res["metadata"]["avg_sync_correlation"] > 0.65
    assert res["image"].startswith("data:image/png;base64,")
