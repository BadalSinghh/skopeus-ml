# Skopeus ML Microservice Deployment Guide (Render)

## 1. Overview
The Skopeus ML Microservice is designed as a standalone container-ready FastAPI application providing deep neural denoising (DnCNN-1D) and NOAA APT demodulation. It serves inference requests from the Skopeus web cockpit on Vercel (`https://skopeus.vercel.app`).

---

## 2. Hardware and Acceleration Model
* **Auto-Hardware Detection:** The service checks `torch.cuda.is_available()`.
  * If CUDA is present (e.g. NVIDIA RTX GPU), it runs parallelized batch tensor operations on the GPU device.
  * If CUDA is unavailable (e.g. Render Free/Standard CPU instance), it safely falls back to high-performance CPU inference via PyTorch CPU runtime without degrading image quality or accuracy.
* **No Artificial Claims:** The `/health` and `/api/process` endpoints report the actual physical compute device (`"CPU"` or `"CUDA:0"`) in telemetry headers and response JSON.

---

## 3. Render Deployment Setup

### Option A: Deploy via `render.yaml` Blueprint (Recommended)
The repository contains a certified `render.yaml` specification:
```yaml
services:
  - type: web
    name: skopeus-ml
    env: python
    region: oregon
    plan: free
    healthCheckPath: /health
    buildCommand: pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: ALLOWED_ORIGINS
        value: "https://skopeus.vercel.app,http://localhost:5173"
      - key: PING_INTERVAL_SECONDS
        value: "600"
```

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** &rarr; **Blueprint**.
3. Connect your repository `https://github.com/BadalSinghh/skopeus-ml`.
4. Render automatically parses `render.yaml` and provisions the web service.

### Option B: Manual Web Service Creation
1. Go to **New +** &rarr; **Web Service**.
2. Connect `BadalSinghh/skopeus-ml`.
3. Configure the build parameters:
   * **Runtime:** Python
   * **Build Command:** `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
   * **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   * **Health Check Path:** `/health`
4. Set Environment Variables:
   * `PYTHON_VERSION`: `3.11.9`
   * `ALLOWED_ORIGINS`: `https://skopeus.vercel.app,http://localhost:5173`
   * `PING_INTERVAL_SECONDS`: `600`
5. Click **Create Web Service**.

---

## 4. Cold-Start Mitigation (Render Free Tier)
Render's free tier spins down web instances after 15 minutes of inactivity.
* `ping_service.py` runs an asynchronous background ping bot on startup.
* When `RENDER_EXTERNAL_URL` is detected (automatically supplied by Render), the bot sends a lightweight HTTP GET to its own `/api/ping` endpoint every 10 minutes (600s), maintaining warm memory cache for fast sub-second inference for frontend users.

---

## 5. Local Development
```bash
# Activate python environment
cd skopeus-ml
pip install -r requirements.txt

# Start local server on port 8000
python app.py
# Or with uvicorn:
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Test endpoints:
* `curl http://127.0.0.1:8000/health`
* Swagger UI: `http://127.0.0.1:8000/docs`
