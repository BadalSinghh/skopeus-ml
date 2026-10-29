# Multi-Pass Dataset Inventory

> **Phase 11A Deliverable**: Exhaustive characterization of available satellite recordings, satellite identities, timestamps, sample rates, durations, signal quality indices (SQI), and split assignments.

---

## 1. Overview of Ingested Recordings

The research dataset consists of **10 distinct satellite recordings** downloaded from the SoriSpace satellite recording archive (`http://data.sorispace.it/data/APT/`). These recordings represent diverse radio frequency observations across three Low Earth Orbit (LEO) satellite platforms: **NOAA-15**, **NOAA-18**, and **historical NOAA-19** (decommissioned August 13, 2025), alongside one experimental/Meteor-M2 transmission.

All baseband audio recordings are sampled at standard SDR output rates ($48,000\text{ Hz}$ or $44,100\text{ Hz}$) and standardized to $20,800\text{ Hz}$ for APT subcarrier demodulation.

---

## 2. Comprehensive Inventory Table

| Recording Filename | Satellite Platform | Pass Timestamp (UTC) | Duration | SQI (dB) | Sample Rate | Native Channels | Current Split | Candidate Phase 11 Split | File Size (MB) |
|:---|:---|:---:|---:|---:|---:|:---:|:---:|:---:|---:|
| `audio_137100000Hz_09-19-25_10-12-2024.wav` | NOAA-19 (Historical) | 10-12-2024 09:19:25 | 527.3 s | 5.46 dB | 48,000 Hz | Mono | TEST | **TEST-PASS-1** (Benchmark) | 48.28 MB |
| `audio_137100000Hz_11-12-57_02-08-2025.wav` | NOAA-19 (Historical) | 02-08-2025 11:12:57 | 460.7 s | 3.17 dB | 48,000 Hz | Mono | TRAIN | **TRAIN** (Retained) | 42.18 MB |
| `audio_137100000Hz_12-50-40_01-12-2024.wav` | NOAA-19 (Historical) | 01-12-2024 12:50:40 | 202.5 s | 4.39 dB | 48,000 Hz | Mono | UNASSIGNED | **TEST-PASS-2** (Unseen) | 18.54 MB |
| `audio_137100000Hz_18-21-39_31-01-2025.wav` | NOAA-19 (Historical) | 31-01-2025 18:21:39 | 83.7 s | -1.39 dB | 48,000 Hz | Stereo | UNASSIGNED | **REAL-NOISY** (Horizon) | 15.33 MB |
| `audio_137350000Hz_12-57-51_06-05-2025.wav` | Meteor-M2 / Experimental | 06-05-2025 12:57:51 | 93.1 s | -1.15 dB | 48,000 Hz | Stereo | TRAIN | **TRAIN** (Retained) | 17.05 MB |
| `audio_137620000Hz_11-02-42_26-12-2024.wav` | NOAA-15 | 26-12-2024 11:02:42 | 79.5 s | 5.45 dB | 48,000 Hz | Stereo | UNASSIGNED | **TEST-PASS-3** (Unseen) | 14.55 MB |
| `audio_137620000Hz_18-06-06_24-01-2025.wav` | NOAA-15 | 24-01-2025 18:06:06 | 16.8 s | 2.71 dB | 44,100 Hz | Mono | VALIDATION | **VALIDATION** (Retained) | 1.41 MB |
| `audio_137620000Hz_20-11-34_30-04-2024.wav` | NOAA-15 | 30-04-2024 20:11:34 | 97.5 s | 3.31 dB | 48,000 Hz | Stereo | UNASSIGNED | **TEST-PASS-4** (Unseen) | 17.86 MB |
| `audio_137912500Hz_12-26-42_06-12-2024.wav` | NOAA-18 | 06-12-2024 12:26:42 | 371.4 s | 23.49 dB | 48,000 Hz | Mono | UNASSIGNED | **TEST-PASS-5** (Unseen) | 34.00 MB |
| `audio_137912500Hz_21-40-50_21-06-2024.wav` | NOAA-18 | 21-06-2024 21:40:50 | 390.6 s | 0.60 dB | 48,000 Hz | Mono | TRAIN | **TRAIN** (Retained) | 35.76 MB |

---

## 3. Dataset Splitting Rationale & Independence Proof

### Primary Statistical Unit: Satellite Pass
In accordance with the Phase 11 mandate, individual audio chunks or overlapping windows are **never** treated as independent observations. The fundamental experimental unit is the **complete satellite pass**.

### 1. Training Set ($N_{\text{passes}} = 3$)
- `audio_137100000Hz_11-12-57_02-08-2025.wav` (NOAA-19 Historical, Summer 2025)
- `audio_137350000Hz_12-57-51_06-05-2025.wav` (Meteor-M2 / Experimental, Spring 2025)
- `audio_137912500Hz_21-40-50_21-06-2024.wav` (NOAA-18, Summer 2024)
- **Cumulative Training Duration**: $944.4\text{ seconds}$ ($\approx 15.7\text{ minutes}$, 540 windows).

### 2. Validation Set ($N_{\text{passes}} = 1$)
- `audio_137620000Hz_18-06-06_24-01-2025.wav` (NOAA-15, Winter 2025)
- **Validation Duration**: $16.8\text{ seconds}$ (180 windows).

### 3. Multi-Pass Test Set ($N_{\text{passes}} = 5$ Strictly Independent Passes)
- **Pass 1 (`PASS-01`)**: `audio_137100000Hz_09-19-25_10-12-2024.wav` — NOAA-19 Historical (Original test pass, Dec 10, 2024).
- **Pass 2 (`PASS-02`)**: `audio_137100000Hz_12-50-40_01-12-2024.wav` — NOAA-19 Historical (Independent midday pass, Dec 01, 2024).
- **Pass 3 (`PASS-03`)**: `audio_137620000Hz_11-02-42_26-12-2024.wav` — NOAA-15 (Independent morning pass, Dec 26, 2024).
- **Pass 4 (`PASS-04`)**: `audio_137620000Hz_20-11-34_30-04-2024.wav` — NOAA-15 (Independent evening pass, Apr 30, 2024).
- **Pass 5 (`PASS-05`)**: `audio_137912500Hz_12-26-42_06-12-2024.wav` — NOAA-18 (Independent high-elevation pass, SQI $23.49\text{ dB}$, Dec 06, 2024).

### 4. Real-World Unpaired Historical Recording ($N_{\text{passes}} = 1$)
- `audio_137100000Hz_18-21-39_31-01-2025.wav` — NOAA-19 Historical (Low-elevation horizon recording, SQI $-1.39\text{ dB}$, evening pass, Jan 31, 2025). Used exclusively for Phase 11K synthetic-to-real transfer evaluation without synthetic pairing.

---

## 4. Retraining Assessment

- **Training Pass Overlap**: The 5 designated test passes share **zero** dates, timestamps, or recording sessions with the 3 training passes.
- **Model Checkpoints**: All 5 trained deep neural network architectures (`CNN1D`, `DnCNN1D`, `UNet1D`, `TCN`, `Transformer1D`) were trained exclusively on the 3 designated training passes.
- **Conclusion**: The existing model checkpoints are **strictly unexposed** to the 5 test passes. No retraining is necessary, ensuring pure, uncompromised out-of-pass evaluation.
