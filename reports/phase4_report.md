# Phase 4 Research Report: DnCNN-1D and 1D U-Net Deep Denoising Architectures

**Project:** Deep Learning Based Noise Removal for LEO Satellite Signals  
**Target downlink:** NOAA POES APT (137.1 / 137.62 / 137.9125 MHz, 2400 Hz AM Subcarrier)  
**Execution Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM), Windows 11, PyTorch 2.11.0+cu128  
**Date:** October 2026  

---

## 1. Executive Summary

In Phase 4, we implemented, optimized, and rigorously evaluated two advanced deep learning architectures for raw 1D satellite signal denoising:
1. **DnCNN-1D (17 layers, 309,825 parameters):** Deep residual convolutional network with Batch Normalization adapted from the Zhang et al. (IEEE TIP 2017) image denoising formulation to 1D continuous baseband audio.
2. **1D U-Net (2,512,497 parameters):** Hierarchical 4-stage multi-scale encoder-decoder architecture with skip connections and linear interpolation upsampling (preventing high-frequency audio checkerboard distortion).

Both models were trained on 540 paired training segments and validated on 180 validation segments using mixed-precision FP16 on the local RTX 3050 GPU. Both models were then benchmarked against our previous classical baselines (No Processing, Butterworth Bandpass Filter, Spectral Gating) and the 1D CNN baseline on **180 completely unseen test segments** drawn from disjoint satellite passes.

### Benchmark Highlights on Unseen Test Passes:
- **1D U-Net** achieved the **highest SNR improvement (+2.27 dB)**, reaching an output SNR of **13.46 dB** and the highest Scale-Invariant Signal-to-Distortion Ratio (**SI-SDR: 15.41 dB**), outperforming linear bandpass filtering (+0.68 dB) by more than $3.3\times$.
- **DnCNN-1D (17L)** achieved the **lowest point-wise error (RMSE: 0.03163, MAE: 0.02393)**—a **39.6% reduction in RMSE** over the unprocessed input—and the highest waveform correlation (**$r = 0.9720$**), while requiring only **2.65 ms** inference latency.
- Peak VRAM footprint during training remained strictly under **160 MB**, proving high training efficiency on consumer laptop GPU hardware.
- Full test suite expanded to **30/30 passing unit tests**.

---

## 2. Architectural Design and Formulation

### 2.1 DnCNN-1D: Deep Residual Denoising Network
In continuous time-domain baseband signals, local carrier periodicity and impulsive interference can be separated effectively through deep residual learning. Following Zhang et al., the network is trained to predict the residual degradation:
$$\hat{\mathbf{v}} = \mathcal{R}(\mathbf{y})$$
$$\hat{\mathbf{x}} = \mathbf{y} - \hat{\mathbf{v}}$$
where $\mathbf{y}$ is the noisy received signal, $\hat{\mathbf{v}}$ is the predicted noise residual, and $\hat{\mathbf{x}}$ is the reconstructed reference signal.

```
Input: y [B, 1, 4096]
  │
  ├──> Conv1D (1 -> 64, K=5, pad=2) + ReLU                         [Layer 1]
  │       │
  │     [Conv1D (64 -> 64, K=5, pad=2) + BatchNorm1d + ReLU] x 15   [Layers 2..16]
  │       │
  │     Conv1D (64 -> 1, K=5, pad=2)                               [Layer 17]
  │       │
  │       v
  │     Residual Noise Prediction: v_hat
  │       │
  └───(-)◄┘
       │
     Output: x_hat = y - v_hat [B, 1, 4096]
```

- **Receptive Field:** With 17 layers and kernel size $K=5$, the effective temporal receptive field is $1 + 16 \times 4 = 65$ samples. At $f_s = 20,800\text{ Hz}$, this corresponds to $3.125\text{ ms}$ or $\approx 7.5$ cycles of the 2400 Hz AM subcarrier, allowing the network to capture instantaneous carrier phase and amplitude envelopes while identifying localized impulse spikes.

### 2.2 1D U-Net: Multi-Scale Hierarchical Encoder-Decoder
While DnCNN maintains constant spatial resolution across all layers, the 1D U-Net captures hierarchical multi-scale temporal context:

```
Input y [B, 1, 4096]
  │
  ├─> Stage 1: Conv1D (1 -> 16, K=7) [B, 16, 4096] ───────────────(Skip 1)────────────┐
  │     ↓ Stride-2 Conv [B, 16, 2048]                                                 │
  ├─> Stage 2: Conv1D (16 -> 32, K=7) [B, 32, 2048] ────────────(Skip 2)─────────┐   │
  │     ↓ Stride-2 Conv [B, 32, 1024]                                             │   │
  ├─> Stage 3: Conv1D (32 -> 64, K=7) [B, 64, 1024] ─────────(Skip 3)──────┐     │   │
  │     ↓ Stride-2 Conv [B, 64, 512]                                         │     │   │
  ├─> Stage 4: Conv1D (64 -> 128, K=7) [B, 128, 512] ──────(Skip 4)───┐     │     │   │
  │     ↓ Stride-2 Conv [B, 128, 256]                                 │     │     │   │
  │   Bottleneck: Dilated Conv Blocks (128 -> 256, K=7, dil=[1,2])    │     │     │   │
  │     ↓ Upsample x2 (Linear) + Conv [B, 128, 512]                   │     │     │   │
  └───> Cat(Upsample, Skip 4) -> Dec Block [B, 128, 512] ◄────────────┘     │     │   │
        ↓ Upsample x2 (Linear) + Conv [B, 64, 1024]                         │     │   │
        Cat(Upsample, Skip 3) -> Dec Block [B, 64, 1024] ◄──────────────────┘     │   │
        ↓ Upsample x2 (Linear) + Conv [B, 32, 2048]                               │   │
        Cat(Upsample, Skip 2) -> Dec Block [B, 32, 2048] ◄────────────────────────┘   │
        ↓ Upsample x2 (Linear) + Conv [B, 16, 4096]                                   │
        Cat(Upsample, Skip 1) -> Dec Block [B, 16, 4096] ◄────────────────────────────┘
        ↓ Head: Conv1d (16 -> 8 -> 1)
      Residual Subtraction: x_hat = y - Head(y) [B, 1, 4096]
```

- **Linear Upsampling vs Transposed Convolutions:** Standard transposed 1D convolutions (`ConvTranspose1d`) are known to produce audible checkerboard artifacts in audio/baseband waveforms due to uneven sample overlap. We employ bilinear interpolation followed by 1D convolution, which guarantees smooth sample reconstruction and stable phase tracking.
- **Skip Connections:** Skip connections propagate high-frequency edge transitions (such as synchronization marker pulses and carrier phase information) directly across scales, preventing temporal smearing.

---

## 3. Training Dynamics and Resource Utilization

Both models were trained using PyTorch Automatic Mixed Precision (AMP FP16) and AdamW with learning rate warmup and Plateau decay on the RTX 3050 Laptop GPU.

| Parameter | 1D CNN Baseline | DnCNN-1D (17L) | 1D U-Net |
|:---|:---:|:---:|:---:|
| **Trainable Parameters** | 65,665 | 309,825 | 2,512,497 |
| **Model Size on Disk** | 0.81 MB | 3.80 MB | 30.25 MB |
| **Batch Size** | 8 | 8 | 8 |
| **Initial Learning Rate** | $1.0 \times 10^{-3}$ | $5.0 \times 10^{-4}$ | $3.0 \times 10^{-4}$ |
| **Optimizer** | AdamW | AdamW | AdamW |
| **Loss Function** | MSE | MSE | MSE |
| **Training Epochs** | 10 | 15 | 15 |
| **Wallclock Time** | 20.7 s | 32.2 s | 35.3 s |
| **Throughput (samples/s)** | ~260 | ~251 | ~229 |
| **Peak GPU VRAM** | 85 MB | 140 MB | 155 MB |
| **Initial Train Loss** | 0.089300 | 0.011044 | 0.006263 |
| **Final Train Loss** | 0.001169 | 0.000848 | 0.001156 |
| **Best Validation Loss** | 0.007672 | 0.004783 | 0.005220 |

---

## 4. Empirical Evaluation Results

Evaluation was performed across all **180 held-out test segments** generated from satellite passes never seen during training.

### 4.1 Quantitative Comparison Matrix

| Method | Type | In SNR (dB) | Out SNR (dB) | $\Delta\text{SNR}$ (dB) | RMSE | MAE | Waveform Correlation $r$ | SI-SDR (dB) | Spectral Conv | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **No Processing (Raw)** | Reference | 11.19 | 11.19 | +0.00 | 0.05235 | 0.03516 | 0.9293 | 13.66 | 0.3601 | 0.00 |
| **Bandpass Filter (1.0–3.8 kHz)** | Classical DSP | 11.19 | 11.88 | +0.68 | 0.03930 | 0.02997 | 0.9615 | 13.35 | 0.2777 | 0.24 |
| **Spectral Gating (STFT)** | Classical DSP | 11.19 | 2.16 | -9.03 | 0.10953 | 0.09415 | 0.5468 | -4.51 | 0.7962 | 1.21 |
| **1D CNN Baseline (5L)** | Deep Learning | 11.19 | 13.09 | +1.89 | 0.03327 | 0.02525 | 0.9711 | 15.33 | 0.2274 | 2.12 |
| **DnCNN-1D (17L)** | Deep Learning | 11.19 | 13.05 | +1.86 | **0.03163** | **0.02393** | **0.9720** | 14.50 | 0.2163 | **2.65** |
| **1D U-Net (4-Stage)** | Deep Learning | 11.19 | **13.46** | **+2.27** | 0.03262 | 0.02430 | 0.9719 | **15.41** | **0.2143** | 5.20 |

---

## 5. Key Findings and Comparative Analysis

### 5.1 1D U-Net Dominates Global SNR and Spectral Fidelity
- **$\Delta\text{SNR} = +2.27\text{ dB}$:** 1D U-Net produced the largest aggregate signal-to-noise ratio enhancement, surpassing both the classical bandpass filter (+0.68 dB) and the 1D CNN (+1.89 dB).
- **Scale-Invariant SDR (15.41 dB):** The U-Net preserves the full dynamic range of amplitude modulation without scaling distortion or phase cancellation.
- **Spectral Convergence (0.2143):** Multi-scale feature extraction allows the U-Net to simultaneously eliminate out-of-band noise (below 1000 Hz and above 3800 Hz) while cleaning intra-band interference around the 2400 Hz carrier without carving notches into the sidebands.

### 5.2 DnCNN-1D Excels at Point-Wise Reconstruction and Impulse Suppression
- **Lowest RMSE (0.03163) and MAE (0.02393):** DnCNN-1D achieved the lowest pointwise error across all 180 test segments, reducing RMSE by **39.6%** compared to the uncleaned raw signal.
- **Highest Correlation ($r = 0.9720$):** Its uniform 17-layer receptive field tracks the fine temporal phase of each cycle of the 2400 Hz subcarrier without the spatial interpolation smoothing inherent in encoder-decoder bottlenecks.
- **Inference Efficiency:** With 309k parameters, DnCNN-1D evaluates a 4096-sample window (196.9 ms of audio) in just **2.65 ms** on the GPU—**74x faster than real time**!

### 5.3 Failure of Spectral Gating on Continuous AM Downlinks
- Spectral gating relies on short-time Fourier thresholding, assuming noise is stationary between speech pauses. In satellite APT downlinks, the 2400 Hz subcarrier is transmitted **continuously** with no silent frames. Consequently, spectral subtraction carves into the AM sidebands and creates severe distortion ($\Delta\text{SNR} = -9.03\text{ dB}$). Deep learning models, by contrast, learn the physical carrier structure and filter noise without destroying modulation.

---

## 6. Artifacts Generated

1. **`src/models/dncnn1d.py`**: Full 1D DnCNN implementation with residual learning and Batch/GroupNorm support.
2. **`src/models/unet1d.py`**: Full 1D U-Net implementation with multi-scale skip connections and linear upsampling.
3. **`configs/model_dncnn.yaml` & `configs/model_unet.yaml`**: Training and architectural configuration files.
4. **`outputs/checkpoints/`**:
   - `dncnn1d_baseline_best.pt` (3.8 MB)
   - `unet1d_baseline_best.pt` (30.2 MB)
5. **`outputs/plots/phase4_model_comparison.png`**: Multi-panel visualization containing waveform overlays, spectrogram comparisons, error residual traces, and test SNR delta bar charts.
6. **`outputs/reports/phase4_model_comparison.json`**: Complete quantitative evaluation metrics across all test segments.
7. **`tests/test_phase4.py`**: Unit test suite verifying tensor dimensionality, norm modes, parameter scaling, and small-batch overfit convergence.

---

## 7. Next Steps: Phase 5 (TCN + Transformer)

With the convolutional baselines firmly established, the next phase will introduce sequence-modeling architectures:
1. **Temporal Convolutional Network (TCN):** Dilated causal/non-causal 1D convolutions with residual blocks to capture long-range temporal dependencies.
2. **1D Transformer Denoiser:** Multi-head self-attention operating over temporal patches to model non-local carrier correlations and wideband interference patterns.
3. Benchmark TCN and Transformer against DnCNN-1D and 1D U-Net on the same test split.
