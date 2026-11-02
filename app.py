"""Skopeus ML Production API Server.

Serves the certified DnCNN-1D neural audio denoiser and NOAA APT satellite image decoder.
Configured for high-throughput deployment on Render, connecting seamlessly to Vercel frontends.
"""

import base64
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Optional

# Limit OpenMP/MKL thread contention on containerized cloud CPUs (e.g. Render 0.1 vCPU)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import soundfile as sf

from apt_service import (
    APT_TARGET_SR,
    decode_apt_audio,
    decode_apt_wav,
)
from denoise_service import (
    CERTIFIED_SHA256,
    denoise_wav,
    get_denoise_service,
)
from ping_service import ping_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("skopeus.api")

SERVER_START_TIME = time.time()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB guard


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: warmup model and start keepalive ping bot on startup."""
    logger.info("Initializing Skopeus ML Backend...")
    try:
        # Pre-warm model into memory on startup
        svc = get_denoise_service(device="auto")
        logger.info("DnCNN-1D model warmed up on device: %s", svc.device)
    except Exception as e:
        logger.error("Failed to pre-warm DnCNN-1D: %s", e)

    # Start Render keepalive ping bot
    ping_bot.start()
    yield
    # Cleanup on shutdown
    ping_bot.stop()
    logger.info("Skopeus ML Backend shutdown complete.")


app = FastAPI(
    title="Skopeus ML Neural Satellite Processing API",
    description="Deep Learning Audio Denoising (DnCNN-1D) & NOAA APT Satellite Image Decoder Microservice.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Vercel and local development
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS") or os.environ.get("CORS_ORIGINS", "")
if allowed_origins_env == "*":
    origins = ["*"]
elif allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    origins = [
        "https://skopeus.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Input-Sample-Rate",
        "X-Input-Channels",
        "X-Output-Sample-Rate",
        "X-Output-Channels",
        "X-Duration-Seconds",
        "X-Device",
        "X-Processing-Time-Seconds",
        "X-Throughput-Realtime-X",
        "X-Checkpoint-Sha256",
    ],
)


def _validate_wav_bytes(raw_bytes: bytes) -> None:
    """Validate standard RIFF/WAVE header."""
    if len(raw_bytes) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload too small to be a valid WAV file.",
        )
    if not (raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio format: Missing RIFF/WAVE container header.",
        )


def _process_audio_file(
    input_wav_path: Path,
    original_name: str,
    device: str = "auto",
    decode: bool = False,
) -> Any:
    """Unified audio processing pipeline."""
    try:
        info = sf.info(str(input_wav_path))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed audio stream: {e}",
        )

    download_name = f"denoised_{Path(original_name).stem}.wav"
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_wav_path = Path(tmp_out.name)
    tmp_out.close()

    try:
        stats = denoise_wav(
            input_wav_path=input_wav_path,
            output_wav_path=output_wav_path,
            device=device,
        )

        with open(output_wav_path, "rb") as f:
            cleaned_bytes = f.read()

        # If decode is requested, run NOAA APT Decoder on the cleaned 20.8 kHz audio
        if decode:
            decode_res = decode_apt_wav(output_wav_path)
            audio_b64 = base64.b64encode(cleaned_bytes).decode("ascii")

            return JSONResponse({
                "success": True,
                "audio_metadata": {
                    "filename": download_name,
                    "input_sample_rate": stats.get("input_sample_rate", info.samplerate),
                    "input_channels": stats.get("input_channels", info.channels),
                    "output_sample_rate": stats.get("output_sample_rate", 20800),
                    "output_channels": 1,
                    "duration_seconds": stats.get("output_duration_sec", round(info.duration, 3)),
                    "device": str(stats.get("device", "AUTO")),
                    "processing_time_seconds": stats.get("processing_time_sec", 0.0),
                    "throughput_realtime_x": stats.get("throughput_realtime_x", 0.0),
                    "checkpoint_sha256": str(stats.get("checkpoint_sha256", CERTIFIED_SHA256)),
                },
                "audio_data_url": f"data:audio/wav;base64,{audio_b64}",
                "image": decode_res.get("image"),
                "channel_a": decode_res.get("channel_a"),
                "channel_b": decode_res.get("channel_b"),
                "metadata": decode_res.get("metadata", {}),
            })

        # Standard binary response
        headers = {
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Input-Sample-Rate": str(stats.get("input_sample_rate", info.samplerate)),
            "X-Input-Channels": str(stats.get("input_channels", info.channels)),
            "X-Output-Sample-Rate": str(stats.get("output_sample_rate", 20800)),
            "X-Output-Channels": str(stats.get("output_channels", 1)),
            "X-Duration-Seconds": str(stats.get("output_duration_sec", round(info.duration, 3))),
            "X-Device": str(stats.get("device", "AUTO")),
            "X-Processing-Time-Seconds": str(stats.get("processing_time_sec", 0.0)),
            "X-Throughput-Realtime-X": str(stats.get("throughput_realtime_x", 0.0)),
            "X-Checkpoint-Sha256": str(stats.get("checkpoint_sha256", CERTIFIED_SHA256)),
        }

        return Response(
            content=cleaned_bytes,
            media_type="audio/wav",
            headers=headers,
        )
    finally:
        if output_wav_path.exists():
            try:
                os.remove(output_wav_path)
            except OSError:
                pass


@app.get("/")
def root():
    """Welcome and API index."""
    return {
        "service": "Skopeus ML Satellite Denoising & APT Decoding Backend",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "ping": "/api/ping",
        "endpoints": {
            "process": "POST /api/process",
            "upload": "POST /api/denoise/upload",
            "capture": "POST /api/denoise/capture",
            "decode": "POST /api/decode/apt",
        },
    }


@app.get("/health")
@app.get("/api/health")
def health():
    """Health check endpoint exposing service, model, and device status."""
    uptime = time.time() - SERVER_START_TIME
    try:
        service = get_denoise_service()
        model_ready = service.model is not None
        device_str = str(service.device).upper()
    except Exception as e:
        model_ready = False
        device_str = f"ERROR: {e}"

    import torch
    cuda_avail = torch.cuda.is_available()

    return {
        "status": "ok" if model_ready else "degraded",
        "service": "Skopeus ML API",
        "uptime_seconds": round(uptime, 2),
        "model_loaded": model_ready,
        "model_architecture": "DnCNN-1D (17 layers, 64 hidden channels, kernel 5)",
        "parameters": 309825,
        "checkpoint": "dncnn1d_production.pt",
        "checkpoint_sha256": CERTIFIED_SHA256,
        "model_verified": model_ready,
        "device": device_str,
        "cuda_available": cuda_avail,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "decoder_available": True,
        "torch_version": torch.__version__,
        "pytorch_version": torch.__version__,
        "ping_bot": ping_bot.get_status(),
    }


@app.get("/api/ping")
def ping():
    """Lightweight keepalive endpoint for Render free tier."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
    }


@app.post("/api/process")
async def process_audio_endpoint(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    device: str = "auto",
):
    """Primary production pipeline endpoint:
    Ingests WAV audio from Skopeus recorder, applies 20.8 kHz preprocessing,
    runs DnCNN-1D neural denoising, decodes NOAA APT imagery, and returns
    full telemetry, cleaned audio (WAV data URL), and rasterized channel images.
    """
    content_type = request.headers.get("content-type", "")
    raw_bytes: bytes = b""
    filename = "recording.wav"

    if "multipart/form-data" in content_type:
        upload = audio or file or audio_file
        if upload is None:
            form = await request.form()
            upload = form.get("audio") or form.get("file") or form.get("audio_file")
        if upload is not None and hasattr(upload, "read"):
            raw_bytes = await upload.read()
            filename = getattr(upload, "filename", "recording.wav") or "recording.wav"
    else:
        raw_bytes = await request.body()
        filename = "recording.wav"

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio payload provided. Send WAV via multipart 'audio' field or request body.",
        )

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio payload exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024*1024)} MB).",
        )

    _validate_wav_bytes(raw_bytes)

    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    input_wav_path = Path(tmp_in.name)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_wav_path = Path(tmp_out.name)
    tmp_in.close()
    tmp_out.close()

    try:
        with open(input_wav_path, "wb") as f:
            f.write(raw_bytes)

        try:
            info = sf.info(str(input_wav_path))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed audio stream: {e}",
            )

        # Cloud CPU latency guard (prevents exceeding cloud gateway timeouts on CPU containers)
        max_cloud_sec = float(os.environ.get("MAX_CLOUD_DURATION_SEC", "0"))
        if max_cloud_sec > 0 and info.duration > max_cloud_sec:
            logger.warning(
                "Audio duration (%.2fs) exceeds MAX_CLOUD_DURATION_SEC (%.2fs). Pre-slicing for cloud processing.",
                info.duration,
                max_cloud_sec,
            )
            aud_data, aud_sr = sf.read(str(input_wav_path), frames=int(max_cloud_sec * info.samplerate), dtype="float32")
            sf.write(str(input_wav_path), aud_data, aud_sr, subtype="PCM_16")
            info = sf.info(str(input_wav_path))

        # 1. Run production DnCNN-1D denoising pipeline
        stats = denoise_wav(
            input_wav_path=input_wav_path,
            output_wav_path=output_wav_path,
            device=device,
        )

        with open(output_wav_path, "rb") as f:
            cleaned_bytes = f.read()

        # 2. Run NOAA APT Decoder on the cleaned 20.8 kHz audio
        decode_res = decode_apt_wav(output_wav_path)
        audio_b64 = base64.b64encode(cleaned_bytes).decode("ascii")
        meta = decode_res.get("metadata", {})

        is_decoded = decode_res.get("success", False)

        response_payload = {
            "success": is_decoded,
            "cleaned_audio": {
                "mime_type": "audio/wav",
                "sample_rate": stats.get("output_sample_rate", 20800),
                "duration_seconds": stats.get("output_duration_sec", round(info.duration, 3)),
                "data": f"data:audio/wav;base64,{audio_b64}",
            },
            "imagery": {
                "full": decode_res.get("image"),
                "channel_a": decode_res.get("channel_a"),
                "channel_b": decode_res.get("channel_b"),
            },
            "decoder": {
                "num_lines": meta.get("num_lines", 0),
                "width": meta.get("width", 2080),
                "sync_lock_percentage": meta.get("sync_lock_percentage", 0.0),
                "average_sync_correlation": meta.get("avg_sync_correlation", 0.0),
                "latency_ms": meta.get("decoder_latency_ms", 0.0),
            },
            "processing": {
                "input_sample_rate": stats.get("input_sample_rate", info.samplerate),
                "output_sample_rate": stats.get("output_sample_rate", 20800),
                "input_duration_sec": stats.get("input_duration_sec", round(info.duration, 3)),
                "processing_time_sec": stats.get("processing_time_sec", 0.0),
                "throughput_realtime_x": stats.get("throughput_realtime_x", 0.0),
                "device": str(stats.get("device", "AUTO")),
                "checkpoint_sha256": str(stats.get("checkpoint_sha256", CERTIFIED_SHA256)),
            },
        }

        if not is_decoded and "error" in decode_res:
            response_payload["error"] = decode_res["error"]

        return JSONResponse(response_payload)
    finally:
        if input_wav_path.exists():
            try:
                os.remove(input_wav_path)
            except OSError:
                pass
        if output_wav_path.exists():
            try:
                os.remove(output_wav_path)
            except OSError:
                pass


@app.post("/api/denoise/upload")
async def denoise_upload(
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    device: str = "auto",
    decode: bool = Query(False, description="If true, decode audio into APT imagery and return JSON payload"),
):
    """Upload an arbitrary noisy .wav file and receive cleaned WAV (or JSON when decode=true)."""
    upload = file or audio_file
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file uploaded. Please provide a .wav file via 'file' or 'audio_file' field.",
        )

    orig_name = upload.filename or "recording.wav"
    if not orig_name.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: '{orig_name}'. Only .wav files are accepted.",
        )

    raw_bytes = await upload.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024*1024)} MB).",
        )

    _validate_wav_bytes(raw_bytes)

    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    input_wav_path = Path(tmp_in.name)
    try:
        tmp_in.write(raw_bytes)
        tmp_in.close()

        return _process_audio_file(
            input_wav_path=input_wav_path,
            original_name=orig_name,
            device=device,
            decode=decode,
        )
    finally:
        if input_wav_path.exists():
            try:
                os.remove(input_wav_path)
            except OSError:
                pass


@app.post("/api/denoise/capture")
async def denoise_capture(
    request: Request,
    device: str = "auto",
    decode: bool = Query(False, description="If true, decode audio into APT imagery and return JSON payload"),
):
    """Process live SDR audio capture buffer."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file") or form.get("audio_file") or form.get("capture")
        if uploaded_file is None or not hasattr(uploaded_file, "read"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No capture file found in multipart form payload.",
            )
        raw_bytes = await uploaded_file.read()
        filename = getattr(uploaded_file, "filename", "sdr_capture.wav")
    else:
        raw_bytes = await request.body()
        filename = "sdr_capture.wav"

    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty SDR capture audio payload received.",
        )

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Capture payload exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024*1024)} MB).",
        )

    _validate_wav_bytes(raw_bytes)

    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    input_wav_path = Path(tmp_in.name)
    try:
        tmp_in.write(raw_bytes)
        tmp_in.close()

        return _process_audio_file(
            input_wav_path=input_wav_path,
            original_name=filename,
            device=device,
            decode=decode,
        )
    finally:
        if input_wav_path.exists():
            try:
                os.remove(input_wav_path)
            except OSError:
                pass


@app.post("/api/decode/apt")
async def decode_apt_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """Decode a cleaned 20.8 kHz mono WAV audio file into NOAA APT imagery."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        upload = file or audio_file
        if upload is None:
            form = await request.form()
            upload = form.get("file") or form.get("audio_file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio file provided. Please provide a 20.8 kHz WAV file via 'file' or 'audio_file'.",
            )
        raw_bytes = await upload.read()
    else:
        raw_bytes = await request.body()

    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio payload received.",
        )

    _validate_wav_bytes(raw_bytes)

    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    input_wav_path = Path(tmp_in.name)
    try:
        tmp_in.write(raw_bytes)
        tmp_in.close()

        try:
            info = sf.info(str(input_wav_path))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed audio file: {e}",
            )

        if info.samplerate != APT_TARGET_SR:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"APTDecoder requires standardized {APT_TARGET_SR} Hz audio, got {info.samplerate} Hz. "
                    "Process audio through /api/denoise/upload first."
                ),
            )

        if info.channels != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"APTDecoder requires mono audio (1 channel), got {info.channels} channels.",
            )

        result = decode_apt_wav(input_wav_path)
        response_payload = {
            "success": result["success"],
            "image": result["image"],
            "channel_a": result["channel_a"],
            "channel_b": result["channel_b"],
            "metadata": result["metadata"],
        }
        if not result["success"]:
            response_payload["error"] = result.get("error", "Decoding failed")
        return JSONResponse(response_payload)
    finally:
        if input_wav_path.exists():
            try:
                os.remove(input_wav_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
