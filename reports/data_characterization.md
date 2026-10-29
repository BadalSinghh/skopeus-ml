# Exploratory Data Analysis & Signal Characterization Report
**Project:** Deep Learning Based Noise Removal for LEO Satellite Signals  
**Target Domain:** Low Earth Orbit (LEO) Meteorological Satellite Downlinks (NOAA POES APT)  
**Dataset Source:** SoriSpace Open Data Satellite Archive (`http://data.sorispace.it/?path=/APT`)  
**Phase:** Phase 1 — Data Investigation & Signal Stage Determination  
**Date:** October 2026  

---

## Executive Summary

Before generating paired synthetic datasets or training deep neural networks, this investigation performed an empirical exploratory data analysis (EDA) of the SoriSpace satellite recording archive. By querying the live repository index and downloading a curated representative sample across frequencies and recording dates, we characterized the signal pipeline stage, acoustic and spectral statistics, sample rates, channel formats, and noise profiles.

### Primary Scientific Conclusions
1. **Signal Pipeline Representation:** The recordings are **FM-demodulated baseband audio signals** representing NOAA Automatic Picture Transmission (APT). They are **not raw complex I/Q baseband data**, nor are they pre-decoded pixel images.
2. **Subcarrier Identification:** The spectral analysis confirms the dominant presence of a **2400 Hz amplitude-modulated (AM) subcarrier**, matching the standard NOAA APT specification ($f_c = 2400\text{ Hz} \pm 2\text{ Hz}$, 4160 baud line rate, 2 lines per second).
3. **Format Heterogeneity:** The archive exhibits heterogeneous sample rates (**48,000 Hz** for the majority, but also **44,100 Hz**), mixed channel topologies (predominantly mono, with rare dual-mono stereo files), and occasional malformed/zero-frame files.
4. **Clean Reference Validity Assessment:** **The raw WAV recordings CANNOT be blindly treated as noise-free reference signals.** In real LEO satellite passes, signal-to-noise ratio varies drastically as a function of satellite elevation angle:
   - *Horizon / Acquisition of Signal (AOS) & Loss of Signal (LOS):* SNR drops to **$\approx 0\text{ dB to } 6\text{ dB}$**, with heavy FM discriminator hiss and atmospheric fading.
   - *Overhead / Zenith ($> 45^\circ$ elevation):* Signals exhibit high carrier purity with in-band to out-of-band SNR exceeding **$+23\text{ dB}$ to $+30\text{ dB}$**.
   - *Consequence:* Using uncurated raw files as clean reference ground truth ($x(t)$) would teach neural networks to reproduce real receiver noise. Therefore, the pipeline strategy is modified to implement an **automated Signal Quality Index (SQI) gating** mechanism to extract clean reference segments exclusively from high-elevation pass windows.

---

## 1. Archive Structure & Distribution

The SoriSpace open-data archive hosts both a compressed monolithic archive (`apt.zip`) and individual accessible WAV files.

| Metric | Archive Value | Notes |
| :--- | :--- | :--- |
| **Monolithic Archive (`apt.zip`)** | 14.30 GB (15,357,244,576 bytes) | Single compressed archive |
| **Total Individual WAV Files** | 740 files | Directly accessible via `/data/APT/<filename>` |
| **Total Uncompressed WAV Volume** | 50.73 GB | Avoids 14 GB bulk download bottleneck |
| **File Size Range** | 0.00 MB to 177.77 MB | Mean: 70.21 MB; Median: 61.16 MB |
| **Standard Pass Duration** | ~8 to 15 minutes | Typical LEO ground station visibility window |

### Satellite & Frequency Breakdown
Satellite identity is directly encoded in the recorded center frequencies in the filenames (`audio_<frequency_Hz>_<HH-MM-SS>_<DD-MM-YYYY>.wav`):

| Satellite | Downlink Frequency | Count | Proportion | Signal Standard |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA-15** | 137.6200 MHz | 408 | 55.1% | APT (VHF FM, 2.4 kHz AM subcarrier) |
| **NOAA-19** | 137.1000 MHz | 146 | 19.7% | APT (VHF FM, 2.4 kHz AM subcarrier) |
| **NOAA-18** | 137.9125 MHz | 111 | 15.0% | APT (VHF FM, 2.4 kHz AM subcarrier) |
| **Experimental / Unknown** | 137.3500 MHz | 71 | 9.6% | Non-standard APT / telemetry / off-tune |
| **Other Offsets** | 137.43 – 137.77 MHz | 4 | 0.5% | Special tracking / Doppler tests |

---

## 2. Empirical Sample Inspection & Physical Measurements

A curated set of 5 recordings spanning distinct satellites, pass durations, and formats was downloaded and analyzed:

| Filename | Satellite | SR (Hz) | Channels | Bit Depth | Duration | Peak (dBFS) | RMS (dBFS) | Clipping (%) | Mid-Pass SNR | Subcarrier at 2.4 kHz |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `audio_137100000Hz_09-19-25_10-12-2024.wav` | NOAA-19 | 48,000 | 1 (Mono) | 16-bit | 527.3 s | 0.00 | -12.65 | 0.076% | **+23.2 dB** | **YES** (2402.3 Hz, 94.9% energy) |
| `audio_137100000Hz_12-04-42_24-02-2024.wav` | NOAA-19 | 48,000 | 2 (Stereo) | 16-bit | 0.0 s | N/A | N/A | Corrupt | N/A | **CORRUPT** (0 audio frames) |
| `audio_137350000Hz_12-57-51_06-05-2025.wav` | 137.35 MHz | 48,000 | 2 (Dual) | 16-bit | 93.1 s | 0.00 | -22.80 | 0.076% | **-2.17 dB** | **NO** (Dominant 2742 Hz, non-APT) |
| `audio_137620000Hz_18-06-06_24-01-2025.wav` | NOAA-15 | **44,100** | 1 (Mono) | 16-bit | 16.8 s | 0.00 | -6.46 | 2.239% | **+2.64 dB** | **YES** (2401.0 Hz, weak pass) |
| `audio_137912500Hz_21-40-50_21-06-2024.wav` | NOAA-18 | 48,000 | 1 (Mono) | 16-bit | 390.6 s | 0.00 | -6.70 | 2.936% | **-0.69 dB** | **YES** (2402.3 Hz, noisy/faded) |

---

## 3. Detailed Physical & Spectral Characteristics

### A. Frequency Spectrum & The 2.4 kHz Subcarrier
- In NOAA POES satellites, the AVHRR (Advanced Very High Resolution Radiometer) imagery is transmitted via the Automatic Picture Transmission (APT) analog standard.
- The image video signal amplitude-modulates a **2400 Hz sinusoidal subcarrier** with a maximum modulation index of ~87%. The resulting DSB-AM signal occupies a bandwidth of approximately **$1000\text{ Hz}$ to $3800\text{ Hz}$**.
- This subcarrier is frequency-modulated onto the 137 MHz VHF carrier with a peak deviation of $\approx 17\text{ kHz}$ (34 kHz RF bandwidth).
- In our spectral analysis of `audio_137100000Hz_09-19-25_10-12-2024.wav`, the peak spectral density occurs at exactly **$2402.3\text{ Hz}$**, with $94.96\%$ of total signal energy concentrated in the 2.4 kHz subcarrier band. The $2.3\text{ Hz}$ frequency offset represents residual Doppler shift ($f_d = \frac{v_r}{c} f_0$) and SDR crystal oscillator tolerance.

### B. Channel Topology: Mono vs. Stereo
- Analysis of 2-channel files (such as `audio_137350000Hz_12-57-51_06-05-2025.wav`) reveals a cross-channel Pearson correlation coefficient of:
  $$r = \frac{\sum (ch_0 - \mu_0)(ch_1 - \mu_1)}{\sigma_0 \sigma_1} = 0.99899$$
- **Finding:** The stereo files are **dual-mono audio feeds**, recorded by sound subsystems that duplicated a mono radio output into left and right channels. They are **not in-phase / quadrature ($I/Q$) signals**.
- **Action:** Any multi-channel input must be converted to mono via channel averaging during ingestion: $x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$.

### C. Sample Rate Heterogeneity
- The recordings contain mixed sample rates: predominantly **48,000 Hz** (standard SDR audio sink rate), but also **44,100 Hz** (standard consumer audio card rate).
- **Action:** The preprocessing pipeline must resample all input recordings to a standard canonical sample rate (e.g., $f_s = 20,800\text{ Hz}$ or $11,025\text{ Hz}$ or $48,000\text{ Hz}$). Resampling to $20,800\text{ Hz}$ or $11,025\text{ Hz}$ maintains the full Nyquist bandwidth ($> 4\text{ kHz}$) while reducing neural network sequence length by more than 4×.

### D. Clipping & Dynamic Range
- All inspected non-empty WAV files use **16-bit signed PCM integer format** (`subtype: PCM_16`), normalized to range $[-1.0, 1.0]$.
- Peak levels hit exactly $0.0\text{ dBFS}$ ($1.0$).
- Full-pass recordings show between $0.07\%$ and $2.94\%$ clipped samples caused by audio gain staging during receiver FM demodulation.
- Preprocessing must implement soft headroom scaling (e.g. $-1\text{ dB}$ to $-3\text{ dB}$ margin) before adding synthetic noise to prevent artificial digital overflow wrapping.

---

## 4. Signal Characterization Visualizations

Diagnostic spectral and waveform plots were generated for the representative passes and saved to `outputs/plots/`:

1. **NOAA-19 High-SNR Pass:** `outputs/plots/audio_137100000Hz_09-19-25_10-12-2024_spec.png`
   - *Waveform:* Periodic 2-line/sec synchronization pulse trains clearly visible.
   - *PSD:* Extreme concentration at 2400 Hz ($> 30\text{ dB}$ above noise floor).
   - *Spectrogram:* Pristine horizontal line patterns corresponding to satellite imagery telemetry.
2. **NOAA-18 Faded/Noisy Pass:** `outputs/plots/audio_137912500Hz_21-40-50_21-06-2024_spec.png`
   - *Waveform:* Significant amplitude modulation and noise bursts.
   - *PSD:* Subcarrier visible, but embedded in broadband FM discriminator triangular noise.
3. **137.35 MHz Non-APT Pass:** `outputs/plots/audio_137350000Hz_12-57-51_06-05-2025_spec.png`
   - *Waveform / PSD:* No 2.4 kHz line structure; peak at 2742 Hz. Confirms non-standard source.

---

## 5. Critical Finding: Is "Clean" a Valid Assumption?

### Analysis
Satellite passes in the SoriSpace archive are real, physical VHF antenna receptions, not synthetic studio signals.
- In `audio_137100000Hz_09-19-25_10-12-2024.wav`:
  - At $t = 10\text{ s}$ (low elevation, satellite rising): In-band SNR is only **$6.0\text{ dB}$**.
  - At $t = 250\text{ s}$ (high elevation, overhead): In-band SNR reaches **$+23.2\text{ dB}$ to $+30.0\text{ dB}$**.
  - At $t = 450\text{ s}$ (low elevation, satellite setting): In-band SNR falls to **$12.2\text{ dB}$**.
- In `audio_137912500Hz_21-40-50_21-06-2024.wav`, the overall mid-pass SNR is **$-0.69\text{ dB}$** due to weak reception or severe antenna fading.

### Consequence
Treating whole raw files as "noise-free reference $x(t)$" is scientifically invalid and would severely compromise supervised deep learning:
- The network would be penalized for removing noise that already exists in the "clean" reference target!
- Denoising metrics (SNR improvement, MSE) would become distorted.

### Modified Dataset Strategy
To resolve this rigorously, our pipeline adapts the following strategy:
1. **Automated Signal Quality Indexing (SQI):** Each candidate recording is segmented into candidate windows. A spectral SNR estimator computes in-band carrier power ($1.0 - 3.8\text{ kHz}$) relative to out-of-band noise ($5.0 - 10.0\text{ kHz}$) and checks for the 2.4 kHz line structure.
2. **Pseudo-Clean Reference Mining:** Only segments achieving $\text{SNR} \ge 20\text{ dB}$ are retained as clean reference targets $x(t)$.
3. **Dual-Use of Real Noise:** Segments from horizon passes where $\text{SNR} \le 5\text{ dB}$ are segmented and extracted to form a **Real Recorded Noise Library** ($n_{\text{real}}[t]$). This directly satisfies Phase 6/7 objectives: combining synthetic noise with real recorded satellite receiver hiss!

---

## 6. Recommended Preprocessing Pipeline

Based on empirical data:
1. **Validation & Filtering:** Reject corrupted files (0 frames) and non-APT frequencies (e.g. 137.35 MHz).
2. **Channel Reduction:** Convert multi-channel audio to mono: $x[n] = \frac{1}{C}\sum_c x_c[n]$.
3. **Resampling:** Resample canonical signals to $f_s = 20,800\text{ Hz}$ or $11,025\text{ Hz}$. (At 20,800 Hz, exactly 5 samples span one 4160-baud telemetry symbol, facilitating sync detection).
4. **DC Removal:** Apply high-pass filter ($f_{\text{cutoff}} = 100\text{ Hz}$) or subtract mean: $x[n] = x[n] - \mu_x$.
5. **Quality Gating:** Segment audio into windows of length $L = 4096$ to $16384$ samples; filter for $\text{SQI} \ge 20\text{ dB}$ for reference generation.
6. **Headroom Normalization:** Scale maximum amplitude to $0.85$ ($-1.4\text{ dBFS}$) prior to synthetic noise injection to prevent ADC clipping saturation.

---

## 7. Recommended Synthetic Noise Models

Given that the signal is FM-demodulated baseband audio with a 2400 Hz AM subcarrier:
1. **Additive White Gaussian Noise (AWGN):** Configurable from $-5\text{ dB}$ to $+20\text{ dB}$ SNR.
2. **Coloured / Discriminator Noise:** FM demodulation below threshold produces "triangular" noise with power spectral density $S_n(f) \propto f^2$.
3. **Impulsive Noise:** Simulates atmospheric lightning static and automotive spark plug bursts (Bernoulli-Gaussian process).
4. **Burst / Terrestrial Interference:** Simulates pager transmitters, local VHF land-mobile radio, and unshielded USB dongle hash.
5. **Slow Gain Fading:** Simulates antenna nulls and satellite Faraday rotation ($a(t)x(t)$ with $0.1 - 2\text{ Hz}$ variation).
6. **Clipping:** Simulates over-driven sound card or SDR baseband ADC saturation.
7. **Real Recorded Noise Mixing:** Additive overlay of real horizon noise segments mined from the SoriSpace archive.

---

## 8. Summary Table & Validation Confirmation

| Characterization Parameter | Verified Value |
| :--- | :--- |
| **DATA SOURCE** | SoriSpace Satellite Archive (`http://data.sorispace.it/?path=/APT`) |
| **WAV FORMAT** | RIFF WAVE, 16-bit PCM integer (`PCM_16`), normalized to $[-1.0, 1.0]$ |
| **SAMPLE RATE** | Mixed: 48,000 Hz (dominant) and 44,100 Hz |
| **CHANNELS** | Predominantly Mono (1 channel); rare Dual-Mono Stereo (2 identical channels) |
| **DURATION RANGE** | 16.8 seconds to 527.3 seconds (full pass: ~8.8 minutes) |
| **SIGNAL CHARACTERISTICS** | Real-valued audio; prominent 2400 Hz AM subcarrier; 2-line/sec sync pulses |
| **LIKELY SIGNAL TYPE** | NOAA POES Automatic Picture Transmission (APT) demodulated audio baseband |
| **CLEAN REFERENCE VALID?** | **NO** for full passes; **YES** for high-elevation zenith segments ($\text{SNR} \ge 20\text{ dB}$) |
| **RECOMMENDED PREPROCESSING** | Filter 0-frame files $\rightarrow$ Resample to 20,800 Hz $\rightarrow$ Mono $\rightarrow$ DC removal $\rightarrow$ SQI gating |
| **RECOMMENDED NOISE MODELS** | AWGN, Triangular FM noise, Impulsive bursts, Faraday fading, Real recorded noise |
| **RECOMMENDED NEXT STEP** | Proceed to **Phase 2: Minimal Dataset Pipeline** using vetted clean segments |
