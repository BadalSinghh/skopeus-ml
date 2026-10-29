# Phase 2 & Phase 3 Research Report: Minimal Dataset Pipeline & Baseline Denoising Benchmark
**Project:** Deep Learning Based Noise Removal for LEO Satellite Signals  
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.8, PyTorch AMP)  
**Target:** NOAA POES Automatic Picture Transmission (APT) Demodulated Audio  
**Date:** October 2026  

---

## 1. Phase 2: Minimal Dataset Pipeline Implementation

### A. Pass-Level Disjoint Splitting (Preventing Data Leakage)
As established in the project guidelines, random chunk-level splitting introduces massive data leakage because overlapping or adjacent chunks from the same satellite pass share identical atmospheric fading, Doppler curves, and ground-station telemetry. 

In [`src/data/splits.py`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/src/data/splits.py), we partitioned the entire SoriSpace archive catalog strictly at the **recording / pass level** across satellites (NOAA-15, NOAA-18, NOAA-19) into [`metadata/splits.csv`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/metadata/splits.csv):

| Split Partition | Satellite Passes | Proportion | Contamination Risk |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 466 recordings | 70.1% | Disjoint |
| **VALIDATION** | 100 recordings | 15.0% | Disjoint |
| **TEST (Held-out)** | 99 recordings | 14.9% | Strictly Unseen ($\text{Train} \cap \text{Test} = \emptyset$) |

---

### B. Standardized Non-Destructive Preprocessing
To resolve the sample-rate heterogeneity and channel differences identified in Phase 1, [`src/data/preprocessing.py`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/src/data/preprocessing.py) implements:
1. **Mono Downmixing:** $x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$.
2. **Rational Polyphase Resampling:** Signals are converted to a standard canonical sample rate of **$f_s = 20,800\text{ Hz}$** using `scipy.signal.resample_poly` with FIR anti-aliasing. At 20.8 kHz, exactly 5 samples span each 4160-baud APT telemetry symbol, preserving Nyquist fidelity while reducing sequence length by $56.7\%$ relative to 48 kHz.
3. **DC Bias Removal:** 2nd-order Butterworth high-pass filtering ($f_{\text{cutoff}} = 100\text{ Hz}$).
4. **Headroom Normalization:** Linear peak scaling to $0.85$ ($-1.4\text{ dBFS}$) prevents digital clipping wrap-around when synthetic noise is injected.
5. **Signal Quality Index (SQI) Gating:** Spectral in-band to out-of-band power ratio calculation ($\ge 15.0\text{ dB}$) ensures only high-elevation overhead segments are mined as clean ground-truth reference targets $x(t)$.

---

### C. Realistic Channel Degradation Engine
The modular degradation suite in `src/noise/` implements:
- **AWGN (`awgn.py`):** Gaussian noise calibrated to exact target SNRs from $-5\text{ dB}$ to $+20\text{ dB}$.
- **Impulsive Noise (`impulsive.py`):** Bernoulli-Gaussian process simulating lightning static and ignition noise.
- **Burst Interference (`burst.py`):** Localized high-power VHF interference with Tukey window tapering.
- **Narrowband Interference (`narrowband.py`):** Discrete interfering sinusoidal carriers in the APT passband.
- **Faraday Gain Fading (`fading.py`):** Smooth $a(t)x(t)$ amplitude modulation ($0.5\text{ Hz}$) simulating satellite spin and polarization nulls.
- **SSB Frequency Offset (`frequency_offset.py`):** Real-valued Hilbert transform single-sideband frequency modulation ($x_a(t)e^{j 2\pi \Delta f t}$) shifting the subcarrier by $\Delta f$ without complex-IQ conversion.
- **Receiver Saturation (`clipping.py`):** Hyperbolic tangent soft saturation and hard ADC clamping.
- **Combined Channel (`simulator.py`):** Stochastic composition combining fading, narrowband tones, burst interference, and AWGN background.

---

### D. Dataset Generation & PyTorch Streaming
Executing `python scripts/generate_dataset.py --config configs/data.yaml` produced **900 paired segments** saved in [`metadata/dataset_manifest.csv`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/metadata/dataset_manifest.csv):
- **Window Size:** $L = 4096$ samples ($196.9\text{ ms}$)
- **Hop Size:** $H = 2048$ samples ($50\%$ overlap)
- **Train Set:** 540 paired samples (NOAA-18 & NOAA-19 passes)
- **Validation Set:** 180 paired samples (NOAA-15 pass)
- **Test Set:** 180 paired samples (held-out NOAA-19 pass)
- **Streaming DataLoader:** [`src/data/dataset.py`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/src/data/dataset.py) loads compressed `.npz` files on-demand, streaming batches directly into VRAM on the RTX 3050 with zero system RAM bloat.

---

## 2. Phase 3: Baseline Experiments (Classical DSP vs. 1D CNN)

### A. Baseline Formulations
1. **Baseline 1 — Identity (No Processing):** $y(t) \rightarrow \hat{x}(t) = y(t)$. Represents the uncorrected noisy reception.
2. **Baseline 2 — Bandpass Filter:** 4th-order Butterworth bandpass filter ($1000\text{ Hz} - 3800\text{ Hz}$). Preserves the APT AM subcarrier bandwidth while stripping out-of-band FM discriminator noise.
3. **Baseline 3 — Spectral Gating:** Short-Time Fourier Transform (STFT) adaptive noise floor estimation and sub-band soft thresholding.
4. **Neural Benchmark — 1D CNN Denoiser:** 5-layer 1D convolutional network with GroupNorm, LeakyReLU, and residual noise learning:
   $$\hat{x} = y - \mathcal{F}_{\theta}(y)$$
   - **Parameters:** 65,665 parameters ($0.25\text{ MB}$)
   - **Optimization:** AdamW ($\text{lr} = 10^{-3}$), MSE Loss, ReduceLROnPlateau scheduler.
   - **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU, PyTorch AMP (FP16).

---

### B. Training Dynamics on RTX 3050 GPU
Training for 10 epochs completed in **20.7 seconds** ($\approx 1.1\text{ s / epoch}$):
- **Loss Progression:** Train MSE decreased from **$0.089348$** (Epoch 1) to **$0.001169$** (Epoch 10) — a **$76.4\times$ reduction**.
- **Validation MSE:** Improved from **$0.010158$** to **$0.005461$**.
- **Memory Footprint:** Peak VRAM consumption was only **$85\text{ MB}$**, verifying that the pipeline is lightweight and stable on the laptop GPU.

---

### C. Held-Out Test Set Benchmark Results
All methods were evaluated across **180 unseen test segments** from held-out satellite passes containing diverse noise types (AWGN, Impulsive, Burst, Narrowband, Fading, Combined) across SNRs from $-5\text{ dB}$ to $+20\text{ dB}$:

| Method | Input SNR | Output SNR | $\Delta$ SNR (Gain) | RMSE | MAE | Pearson Correlation | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **No Processing (Raw)** | $11.19\text{ dB}$ | $11.19\text{ dB}$ | $+0.00\text{ dB}$ | $0.05235$ | $0.03516$ | $0.9293$ | $0.00\text{ ms}$ |
| **Bandpass Filter** | $11.19\text{ dB}$ | $11.88\text{ dB}$ | $+0.68\text{ dB}$ | $0.03930$ | $0.02997$ | $0.9615$ | $0.21\text{ ms}$ |
| **Spectral Gating** | $11.19\text{ dB}$ | $2.16\text{ dB}$ | $-9.03\text{ dB}$ | $0.10953$ | $0.09415$ | $0.5468$ | $1.08\text{ ms}$ |
| **1D CNN Denoiser** | $11.19\text{ dB}$ | **$13.09\text{ dB}$** | **$+1.89\text{ dB}$** | **$0.03327$** | **$0.02525$** | **$0.9711$** | $1.97\text{ ms}$ |

---

### D. Key Empirical Insights
1. **Conventional Filtering Limitations:** The bandpass filter achieves a modest $+0.68\text{ dB}$ SNR improvement by eliminating out-of-band hiss, but is completely incapable of removing in-band noise overlapping the $2400\text{ Hz}$ telemetry subcarrier.
2. **Spectral Gating Failure:** Spectral gating degrades performance drastically ($\Delta\text{SNR} = -9.03\text{ dB}$) because satellite APT telemetry is an analog continuous AM carrier; gating thresholds aggressively carve out low-amplitude scanlines and destroy the carrier continuity.
3. **Deep Learning Superiority:** The 1D CNN denoiser achieves **$+1.89\text{ dB}$ average SNR gain** and reduces RMSE by **$36.4\%$** compared to the raw signal ($0.03327$ vs $0.05235$), preserving the subcarrier structure with a correlation of **$0.9711$**.
4. **Real-Time Efficiency:** At **$1.97\text{ ms}$ per 197 ms window** on the RTX 3050, the model runs at **$100\times$ faster than real-time**, making it suitable for live software-defined radio ground-station reception pipelines.
5. **Visual Verification:** The generated comparison figure [`outputs/plots/baseline_cnn_waveform_spectrogram.png`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/outputs/plots/baseline_cnn_waveform_spectrogram.png) confirms clean reconstruction of the 2400 Hz line structure and elimination of burst/Gaussian artifacts.

---

## 3. Test Suite Verification
The complete test suite ([`tests/test_phase1.py`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/tests/test_phase1.py), [`tests/test_phase2.py`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/tests/test_phase2.py), [`tests/test_phase3.py`](file:///c:/Users/Keshav/Desktop/SKOPEUS%20BADAL/tests/test_phase3.py)) passed **20 out of 20 unit and regression tests** in 7.69 seconds.
