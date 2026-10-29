# Phase 5 Research Report: Sequence Modeling Architectures (TCN vs 1D Transformer)

**Project:** Deep Learning Based Noise Removal for LEO Satellite Signals  
**Target downlink:** NOAA POES APT (137.1 / 137.62 / 137.9125 MHz, 2400 Hz AM Subcarrier)  
**Execution Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM), Windows 11, PyTorch 2.11.0+cu128  
**Date:** October 2026  

---

## 1. Executive Summary

Phase 5 introduced sequence-modeling architectures to capture long-range temporal dependencies and periodic structure in satellite signals:
1. **Temporal Convolutional Network (TCN, 233,473 parameters):** 6-stage dilated residual network with exponentially increasing dilation factors ($d \in \{1, 2, 4, 8, 16, 32\}$) and symmetric non-causal receptive field covering **505 samples** ($\approx 58$ cycles of the 2400 Hz AM carrier).
2. **1D Transformer Denoiser (692,289 parameters):** Patch-based tokenizer ($P=16$ samples/token), sinusoidal positional encoding, 4 Pre-LN Transformer encoder layers (4 heads, $d_{model}=128$), and a transposed-convolutional reconstruction head with phase refinement.

Both models were trained on 540 paired training segments and validated on 180 validation segments using mixed-precision FP16 on the local RTX 3050 Laptop GPU. Both models were then benchmarked alongside the previous models (No Processing, Bandpass, Spectral Gating, 1D CNN, DnCNN-1D, 1D U-Net) across **180 completely unseen test segments** from held-out satellite passes.

### Major Discoveries and Benchmark Highlights:
- **TCN achieved the highest overall denoising performance across the entire project to date:**
  - **$\Delta\text{SNR}$ Improvement:** **+2.42 dB** (Output SNR: **13.62 dB**)
  - **Scale-Invariant SDR:** **16.49 dB** (highest across all 8 methods)
  - **Reconstruction RMSE:** **0.02868** (**-45.2% reduction in RMSE** compared to raw input)
  - **Reconstruction MAE:** **0.02182** (**-37.9% reduction in MAE**)
  - **Waveform Correlation:** **$r = 0.9842$** (highest temporal alignment)
  - **Spectral Convergence:** **0.2003** (lowest spectral distortion)
  - **Inference Latency:** **3.21 ms** on GPU (> $61\times$ faster than real-time)
- **1D Transformer (+1.08 dB $\Delta\text{SNR}$, 14.01 dB SI-SDR)** significantly outperformed classical bandpass filtering (+0.68 dB) and spectral gating (-9.03 dB), but lagged TCN. We analyze why patch-based tokenization introduces subtle subcarrier phase smoothing.
- Full test suite expanded to **39/39 passing unit tests**.

---

## 2. Architectural Design and Formulation

### 2.1 Dilated Temporal Convolutional Network (TCN)
Satellite APT downlinks exhibit strict periodicity dictated by the 2400 Hz AM subcarrier and line-synchronization pulses. Standard CNNs require very deep stacks or strided downsampling to observe multiple carrier cycles. TCN solves this via dilated residual convolutions without reducing temporal resolution:

```
Input y [B, 1, 4096]
  │
  ├──> Conv1D (1 -> 64, K=5, d=1)  + GroupNorm + LeakyReLU + Conv1D + Drop ──(+)─> [B, 64, 4096]  (Stage 1, d=1)
  │                                                                           │
  ├──> Conv1D (64 -> 64, K=5, d=2) + GroupNorm + LeakyReLU + Conv1D + Drop ──(+)─> [B, 64, 4096]  (Stage 2, d=2)
  │                                                                           │
  ├──> Conv1D (64 -> 64, K=5, d=4) + GroupNorm + LeakyReLU + Conv1D + Drop ──(+)─> [B, 64, 4096]  (Stage 3, d=4)
  │                                                                           │
  ├──> Conv1D (64 -> 64, K=5, d=8) + GroupNorm + LeakyReLU + Conv1D + Drop ──(+)─> [B, 64, 4096]  (Stage 4, d=8)
  │                                                                           │
  ├──> Conv1D (64 -> 64, K=5, d=16)+ GroupNorm + LeakyReLU + Conv1D + Drop ──(+)─> [B, 64, 4096]  (Stage 5, d=16)
  │                                                                           │
  ├──> Conv1D (64 -> 64, K=5, d=32)+ GroupNorm + LeakyReLU + Conv1D + Drop ──(+)─> [B, 64, 4096]  (Stage 6, d=32)
  │                                                                           │
  │    Output Projection Head: Conv1d (64 -> 32 -> 1)                         │
  │    Residual Subtraction: x_hat = y - Head(Features)                       │
  └───(-)◄────────────────────────────────────────────────────────────────────┘
       v
     Output x_hat [B, 1, 4096]
```

- **Receptive Field Mathematical Analysis:**
  $$\text{RF} = 1 + 2 \times (K - 1) \sum_{i=0}^{N-1} 2^i = 1 + 2 \times 4 \times (1 + 2 + 4 + 8 + 16 + 32) = 1 + 8 \times 63 = 505 \text{ samples}$$
  At $f_s = 20,800\text{ Hz}$, $505\text{ samples} = 24.28\text{ ms}$. Since $T_{\text{carrier}} = 1 / 2400\text{ Hz} \approx 0.4167\text{ ms}$, the TCN receptive field spans **58.27 consecutive carrier cycles**. This enables the network to accurately reconstruct continuous carrier amplitude modulation through deep bursts and tone interference.

### 2.2 1D Transformer Denoiser
```
Input y [B, 1, 4096]
  │
  ├─> 1D Patch Tokenizer: Conv1D (1 -> 128, K=16, S=16) ──────────> [B, 256, 128]
  │     ↓ LayerNorm + Sinusoidal Positional Encoding
  │   4x Transformer Encoder Layers (4 heads, d_model=128, d_ff=256, Pre-LN)
  │     ↓ Encoded Tokens [B, 256, 128]
  │   Transposed Convolution Decoder: ConvTranspose1d (128 -> 64, K=16, S=16) ──> [B, 64, 4096]
  │     ↓ Phase Refinement Conv1d Block (64 -> 64, K=7, pad=3)
  │   Linear Projection Head: Conv1d (64 -> 1)
  └───(-)◄ Residual Subtraction: x_hat = y - Head(y)
       v
     Output x_hat [B, 1, 4096]
```

- **Patch Tokenization:** By downsampling $4096\text{ samples} \rightarrow 256\text{ tokens}$, self-attention requires only $O(256^2) = 65,536$ operations per head per layer. This enables the Transformer to execute with only **89 MB peak VRAM** and train in under **29 seconds**.
- **The Sub-Cycle Phase Bottleneck:** Because 1 token corresponds to 16 samples, and 1 cycle of 2400 Hz is 8.67 samples, each token spans nearly two full cycles. The Transformer learns token-level inter-burst semantics, but reconstructing the intra-patch cycle phase relies on the transposed convolution upsampler.

---

## 3. Training Dynamics and Resource Utilization

All deep learning models trained on the RTX 3050 Laptop GPU (6.0 GB VRAM):

| Model | Parameters | Disk Size | Epochs | Training Time | Peak VRAM | Best Val Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1D CNN Baseline** | 65,665 | 0.81 MB | 10 | 20.7 s | 85 MB | 0.007672 |
| **DnCNN-1D (17L)** | 309,825 | 3.80 MB | 15 | 32.2 s | 140 MB | 0.004783 |
| **1D U-Net** | 2,512,497 | 30.25 MB | 15 | 35.3 s | 155 MB | 0.005220 |
| **TCN (6-Stage)** | **233,473** | **2.85 MB** | 15 | 40.1 s | 281 MB | **0.004258** |
| **1D Transformer** | 692,289 | 10.48 MB | 15 | **28.9 s** | **89 MB** | 0.005672 |

---

## 4. Comprehensive Benchmark Across All 8 Methods

Evaluated across **180 held-out test segments** generated from satellite passes never observed during training:

| Method | Architecture Type | In SNR (dB) | Out SNR (dB) | $\Delta\text{SNR}$ (dB) | RMSE | MAE | Pearson $r$ | SI-SDR (dB) | Spectral Conv | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **No Processing (Raw)** | Ground Truth Baseline | 11.19 | 11.19 | +0.00 | 0.05235 | 0.03516 | 0.9293 | 13.66 | 0.3601 | 0.00 |
| **Bandpass Filter** | 4th-order Butterworth | 11.19 | 11.88 | +0.68 | 0.03930 | 0.02997 | 0.9615 | 13.35 | 0.2777 | **0.23** |
| **Spectral Gating** | Classical STFT Subtraction | 11.19 | 2.16 | -9.03 | 0.10953 | 0.09415 | 0.5468 | -4.51 | 0.7962 | 1.03 |
| **1D CNN Baseline** | 5-Layer Dilated CNN | 11.19 | 13.09 | +1.89 | 0.03327 | 0.02525 | 0.9711 | 15.33 | 0.2274 | 2.29 |
| **DnCNN-1D (17L)** | Deep Residual CNN | 11.19 | 13.05 | +1.86 | 0.03163 | 0.02393 | 0.9720 | 14.50 | 0.2163 | 2.46 |
| **1D U-Net** | 4-Stage Hierarchical Encoder-Decoder | 11.19 | 13.46 | +2.27 | 0.03262 | 0.02430 | 0.9719 | 15.41 | 0.2143 | 5.64 |
| **TCN (6-Stage)** | **Dilated Residual Sequence Model** | 11.19 | **13.62** | **+2.42** | **0.02868** | **0.02182** | **0.9842** | **16.49** | **0.2003** | 3.21 |
| **1D Transformer** | Patch Multi-Head Self-Attention | 11.19 | 12.27 | +1.08 | 0.03782 | 0.02678 | 0.9575 | 14.01 | 0.2490 | 3.25 |

---

## 5. Architectural Comparison and Key Findings

```
                              THE ACCURACY vs COMPLEXITY PARETO FRONTIER
      
    14.0 dB ──┐                                         ★ TCN (+2.42 dB, 233k params)
              │
    13.5 dB ──┤                                                 ▲ 1D U-Net (+2.27 dB, 2.51M params)
              │                                 ● 1D CNN (+1.89 dB, 65k params)
    13.0 dB ──┤                                         ■ DnCNN (+1.86 dB, 310k params)
  Out SNR     │
    12.5 dB ──┤
              │                                         ◆ 1D Transformer (+1.08 dB, 692k params)
    12.0 dB ──┤                 ◆ Classical Bandpass (+0.68 dB, 0 params)
              │
    11.0 dB ──┤ ── Unprocessed Raw Input (11.19 dB) ─────────────────────────────────────────────
              │
     2.0 dB ──┤ ✕ Classical Spectral Gating (-9.03 dB) [Modulation Destroyer]
              └──────────────────────────────────────────────────────────────────────────────────
                0.0 ms          1.0 ms          2.0 ms          3.0 ms          4.0 ms    5.0+ ms
                                            Inference Latency (RTX 3050 GPU)
```

### 5.1 Why TCN Outperforms All Other Architectures
1. **Zero Spatial Downsampling:** Unlike U-Net and Transformer, which downsample time representations (via stride or patch tokenization) and attempt to reconstruct them, TCN preserves 1:1 temporal sample resolution at all 6 stages. Every sample has a direct residual connection to the input.
2. **58-Cycle Dilated Receptive Field:** Dilations $d = [1, 2, 4, 8, 16, 32]$ enable TCN to compare the current sample with corresponding samples from 58 prior and future carrier cycles. This allows it to cleanly separate coherent subcarrier oscillation from uncorrelated Gaussian, impulsive, or burst noise.
3. **Parameter Efficiency:** TCN achieves the project's highest score (+2.42 dB $\Delta\text{SNR}$, 0.02868 RMSE, 0.9842 correlation) using only **233k parameters**—nearly **$11\times$ fewer parameters** than 1D U-Net (2.51M).

### 5.2 Why 1D Transformer Excels in Speed but Lags in Carrier Phase Fidelity
- The 1D Transformer is remarkably lightweight in memory (89 MB) and fast to train (28.9s).
- However, audio patch tokenization ($P=16$) operates at the token level, meaning self-attention calculates interactions between chunks of 16 samples. At 20.8 kHz, where one subcarrier cycle is only 8.67 samples, the attention heads model token energy and burst envelope transitions rather than sub-sample carrier phase.
- While it comfortably beats classical filters, fine-grain time-series denoising strongly favors architectures with 1:1 sample resolution (TCN and DnCNN).

---

## 6. Artifacts and Code Deliverables

1. **`src/models/tcn.py`**: Full 6-stage dilated residual TCN implementation.
2. **`src/models/transformer.py`**: 1D Transformer denoiser with patch tokenization, sinusoidal PE, and transposed-convolutional refinement head.
3. **`configs/model_tcn.yaml` & `configs/model_transformer.yaml`**: Training and architectural configuration files.
4. **`outputs/checkpoints/`**:
   - `tcn_baseline_best.pt` (2.85 MB)
   - `transformer_baseline_best.pt` (10.48 MB)
5. **`outputs/plots/phase5_model_comparison.png`**: Comprehensive multi-panel visualization showing waveform zooms, error residual traces, full benchmark comparison bar charts, and spectrograms.
6. **`outputs/reports/phase5_model_comparison.json`**: Complete numerical metric records for all 8 methods across the 180 test segments.
7. **`tests/test_phase5.py`**: Unit test suite for sequence modeling architectures (**39/39 passing tests** across project).

---

## 7. Next Steps: Phase 6 & Phase 7/8

With all 5 deep neural architectures (1D CNN, DnCNN-1D, 1D U-Net, TCN, 1D Transformer) implemented, trained, and benchmarked:
1. **Phase 6 & 7: Full Realistic Noise Matrix & Expanded Dataset Generation:**
   - Expand the dataset to include a comprehensive noise matrix (varying SNR levels from -10 dB to +20 dB, burst durations, frequency offset drift, multi-tone interference, and real recorded horizon pass noise).
2. **Phase 9: Satellite Image Decoder Integration:**
   - Decode clean, noisy, and denoised audio through the NOAA APT decoder (`noaa-apt` / image reconstruction module) to evaluate structural SSIM, PSNR, and cloud/land border preservation in the reconstructed satellite imagery.
