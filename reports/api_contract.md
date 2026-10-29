# Skopeus ML Inference & Decoding API Contract

**Version:** 1.0.0  
**Service:** Skopeus ML Production Backend (`skopeus-ml`)  
**Deployment Target:** Render (Web Service)  
**Consumer:** Skopeus Web Cockpit (`skopeus` on Vercel)  
**Protocol:** HTTPS  

---

## 1. Architectural Overview

The Skopeus ML backend is a dedicated Python/FastAPI microservice executing:
1. Standardized 20.8 kHz DSP audio preprocessing (mono downmixing, polyphase resampling, 100 Hz highpass, 0.85 headroom scaling).
2. Certified DnCNN-1D neural audio denoising (17 layers, receptive field 69 samples, 50% overlap-add Hann windowing).
3. Robust NOAA POES APT baseband demodulator & image rasterizer (Hilbert envelope extraction, 1800 Hz lowpass, normalized Sync A detection, +/-50 samples fine line tracking, 2080-pixel raster generation).
4. Direct serialization of 2D weather satellite matrices (Visible Channel A, Thermal IR Channel B, and composite) into Base64 PNG Data URLs and PCM WAV audio Data URLs.

The Skopeus frontend (Vercel) remains a pure client and **never executes ML inference locally**.

---

## 2. Public Endpoints

### 2.1 Health & Diagnostics
* **Path:** `GET /health` (also aliased at `GET /api/health`)
* **Description:** Health check exposing model status, hardware detection, and versions.
* **Authentication:** None required.
* **Response (HTTP 200 OK):**
```json
{
  "status": "ok",
  "service": "Skopeus ML API",
  "uptime_seconds": 128.45,
  "model_loaded": true,
  "model_architecture": "DnCNN-1D (17 layers, 64 hidden channels, kernel 5)",
  "parameters": 309825,
  "checkpoint": "dncnn1d_production.pt",
  "checkpoint_sha256": "24ce759369ad110b23a524bedc2ad90eb608b497801a3f2300861c1fb106a7c2",
  "model_verified": true,
  "device": "CPU",
  "cuda_available": false,
  "gpu_name": "CPU",
  "decoder_available": true,
  "torch_version": "2.x.x",
  "pytorch_version": "2.x.x",
  "ping_bot": {
    "is_running": true,
    "target_url": "https://skopeus-ml.onrender.com/api/ping",
    "ping_count": 12,
    "last_ping_status": "ok",
    "last_ping_time": 1740000000.0,
    "interval_seconds": 600
  }
}
```

---

### 2.2 Primary Production Processing Endpoint
* **Path:** `POST /api/process`
* **Content-Type:** `multipart/form-data`
* **Payload Fields:**
  * `audio`: (Binary File, required) Raw WAV audio file from SDR recording or file upload. Supported sample rates: 8 kHz – 192 kHz (e.g. 48 kHz standard browser SDR recorder or 20.8 kHz standard NOAA APT). Mono and stereo accepted.
  * `device`: (String, optional, default: `"auto"`) Hardware target (`"auto"`, `"cpu"`, or `"cuda"`).

#### Success Response (HTTP 200 OK)
```json
{
  "success": true,
  "cleaned_audio": {
    "mime_type": "audio/wav",
    "sample_rate": 20800,
    "duration_seconds": 16.0,
    "data": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
  },
  "imagery": {
    "full": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "channel_a": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "channel_b": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "decoder": {
    "num_lines": 32,
    "width": 2080,
    "sync_lock_percentage": 100.0,
    "average_sync_correlation": 0.6424,
    "latency_ms": 36.8
  },
  "processing": {
    "input_sample_rate": 48000,
    "output_sample_rate": 20800,
    "input_duration_sec": 16.0,
    "processing_time_sec": 0.35,
    "throughput_realtime_x": 45.7,
    "device": "CPU",
    "checkpoint_sha256": "24ce759369ad110b23a524bedc2ad90eb608b497801a3f2300861c1fb106a7c2"
  }
}
```

#### Degraded / Poor Signal Response (HTTP 200 OK with `success: false`)
When audio is too short (< 0.5 s / 10,400 samples) or synchronization lock cannot be obtained:
```json
{
  "success": false,
  "error": "Insufficient audio duration (< 0.5 s / 10,400 samples) to decode an APT line.",
  "cleaned_audio": {
    "mime_type": "audio/wav",
    "sample_rate": 20800,
    "duration_seconds": 0.3,
    "data": "data:audio/wav;base64,..."
  },
  "imagery": {
    "full": null,
    "channel_a": null,
    "channel_b": null
  },
  "decoder": {
    "num_lines": 0,
    "width": 2080,
    "sync_lock_percentage": 0.0,
    "average_sync_correlation": 0.0,
    "latency_ms": 2.1
  },
  "processing": {
    "input_sample_rate": 48000,
    "output_sample_rate": 20800,
    "input_duration_sec": 0.3,
    "processing_time_sec": 0.04,
    "throughput_realtime_x": 7.5,
    "device": "CPU",
    "checkpoint_sha256": "24ce759369ad110b23a524bedc2ad90eb608b497801a3f2300861c1fb106a7c2"
  }
}
```

#### Error Responses
* **HTTP 400 Bad Request:**
  * Missing file: `{"detail": "No audio payload provided. Send WAV via multipart 'audio' field or request body."}`
  * Non-WAV payload: `{"detail": "Invalid audio format: Missing RIFF/WAVE container header."}`
* **HTTP 413 Payload Too Large:**
  * Audio payload exceeds 50 MB: `{"detail": "Audio payload exceeds maximum allowed size (50 MB)."}`
* **HTTP 500 Internal Server Error:**
  * Denoising engine or decoder unrecoverable failure: `{"detail": "..."}`

---

### 2.3 Render Keepalive Ping
* **Path:** `GET /api/ping`
* **Description:** Low-overhead heartbeat called by the internal background task or external health probes every 600s to prevent Render free-tier sleep.
* **Response (HTTP 200 OK):**
```json
{
  "status": "ok",
  "timestamp": 1740000000.12,
  "uptime_seconds": 128.5
}
```

---

## 3. CORS Policy
Cross-Origin Resource Sharing is configured to allow headers `*` and credentials for:
* `https://skopeus.vercel.app` (Vercel production frontend)
* `http://localhost:5173` (Vite local development)
* `http://localhost:3000` (Alternative local test port)
* Custom origins configured via `ALLOWED_ORIGINS` environment variable (comma-separated list or `*`).
