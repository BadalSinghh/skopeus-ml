# Skopeus ML: Deep Learning Satellite Signal Denoising & NOAA APT Decoder

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Render-Deploy-46E3B7.svg?style=flat&logo=render)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-throughput, cloud-native machine learning microservice for denoising Low Earth Orbit (LEO) satellite communications and decoding real-time weather imagery from the **NOAA POES constellation (NOAA-15, NOAA-18, NOAA-19)**.

Designed for seamless deployment on **Render** (free tier friendly with built-in keep-alive bot) and plug-and-play connection to **Vercel** web applications.

---

## 🛰️ Architecture Overview

```text
       Vercel Frontend (https://skopeus.vercel.app)
                            │
               POST /api/process (WAV file)
                            ▼
       Render ML Microservice (https://skopeus-ml.onrender.com)
  ┌────────────────────────────────────────────────────────┐
  │ 1. Polyphase Resampling to 20,800 Hz mono (exact integer)
  │ 2. 2nd-order Butterworth 100 Hz highpass filter
  │ 3. 0.85 peak headroom scaling
  │ 4. DnCNN-1D Neural Inference (309,825 params, 17 layers)
  │ 5. Weighted overlap-add waveform reconstruction
  │ 6. NOAA APT AM/Hilbert Demodulation & 1800 Hz filter
  │ 7. Normalized cross-correlation Sync A line tracking
  │ 8. 2080 px Composite + 1040 px Channel A/B separation
  └────────────────────────────────────────────────────────┘
                            │
                 Cleaned 20.8 kHz WAV + Decoded PNGs + Sync Telemetry
                            ▼
       Interactive Mission Cockpit (SatDumpPipeline)
```

---

## ⚡ Certified Model Weights

- **Model:** `DnCNN1DDenoiser`
- **Layers:** 17 convolutional layers, 64 feature channels, kernel size 5
- **Parameters:** 309,825
- **Weights File:** `models/dncnn1d_production.pt` (1.2 MB)
- **SHA-256 Checksum:** `24ce759369ad110b23a524bedc2ad90eb608b497801a3f2300861c1fb106a7c2`

---

## 🚀 3-Click Render Deployment Guide

1. Log into [Render.com](https://render.com) and click **New + > Web Service**.
2. Connect your GitHub repository: `BadalSinghh/skopeus-ml`.
3. Configure the following settings:
   - **Environment:** `Python 3`
   - **Build Command:**
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan:** Free
4. Click **Create Web Service**. Your service will be live at `https://skopeus-ml.onrender.com` in ~2 minutes!

> **💡 Built-in Keep-Alive Ping Bot:**  
> Render's free tier automatically suspends web services after 15 minutes of inactivity. This microservice includes an automatic background ping bot (`ping_service.py`) that pings itself every 10 minutes, **keeping your service awake 24/7 with zero cold starts!**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | API status and overview |
| `GET` | `/api/health` | Model status, parameters, uptime, and ping bot diagnostics |
| `GET` | `/api/ping` | Lightweight keepalive endpoint |
| `POST` | `/api/denoise/upload` | Upload `.wav` file $\rightarrow$ returns cleaned 20.8 kHz WAV (`decode=false`) or JSON with audio and satellite imagery (`decode=true`) |
| `POST` | `/api/denoise/capture` | Process live SDR capture buffer |
| `POST` | `/api/decode/apt` | Standalone APT image decoder for cleaned 20.8 kHz audio |

---

## 💻 Local Development & Testing

```bash
# Clone the repository
git clone https://github.com/BadalSinghh/skopeus-ml.git
cd skopeus-ml

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run automated verification tests
pytest tests/ -v

# Start local server
uvicorn app:app --reload --port 8000
```
