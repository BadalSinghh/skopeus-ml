"""Phase 2 Diagnostic Script: Empirical Audit of APT Decoder on Historical NOAA Passes.

Performs:
1. Audit on NOAA-15 pass (audio_137620000Hz_18-06-06_24-01-2025.wav, 44.1 kHz, low SNR)
2. Audit on NOAA-19 pass (audio_137100000Hz_09-19-25_10-12-2024.wav, 48.0 kHz, high SNR)
3. Direct comparison: Raw Native vs Raw 20.8 kHz vs DnCNN-1D Denoised
4. Generates diagnostic figures in outputs/phase2/
"""

import json
import sys
import time
from pathlib import Path

# Add repo root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from server.denoise_service import denoise_wav
from src.evaluation.apt_decoder import APTDecoder, compute_image_psnr, compute_image_ssim
from src.data.preprocessing import resample_audio


def audit_pass(pass_name: str, wav_path: Path, max_frames: int = None, out_prefix: str = "pass"):
    out_dir = Path("outputs/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"AUDITING PASS: {pass_name} ({wav_path.name})")
    print("=" * 80)

    # 1. Read input audio
    if max_frames is not None and max_frames > 0:
        data, orig_sr = sf.read(str(wav_path), frames=max_frames, dtype="float32")
    else:
        data, orig_sr = sf.read(str(wav_path), dtype="float32")
    dur_sec = len(data) / float(orig_sr)
    peak_amp = float(np.max(np.abs(data)))
    rms_amp = float(np.sqrt(np.mean(data**2)))

    # Save test slice
    test_slice_wav = out_dir / f"{out_prefix}_raw.wav"
    sf.write(str(test_slice_wav), data, orig_sr, subtype="PCM_16")

    # 2. Run production DnCNN-1D denoising
    cleaned_wav = out_dir / f"{out_prefix}_cleaned.wav"
    t0 = time.perf_counter()
    denoise_stats = denoise_wav(test_slice_wav, cleaned_wav, device="auto")
    t_denoise = time.perf_counter() - t0

    clean_data, clean_sr = sf.read(str(cleaned_wav), dtype="float32")

    # 3. Decode Raw Native
    t0 = time.perf_counter()
    dec_native = APTDecoder(sr=orig_sr)
    img_native, meta_native = dec_native.decode(data)
    t_dec_native = time.perf_counter() - t0

    # 4. Decode Raw Resampled to 20.8 kHz
    raw_20k = resample_audio(data, orig_sr=orig_sr, target_sr=20800)
    t0 = time.perf_counter()
    dec_20k = APTDecoder(sr=20800)
    img_raw_20k, meta_raw_20k = dec_20k.decode(raw_20k)
    t_dec_raw_20k = time.perf_counter() - t0

    # 5. Decode Cleaned Audio at 20.8 kHz
    t0 = time.perf_counter()
    img_clean, meta_clean = dec_20k.decode(clean_data)
    t_dec_clean = time.perf_counter() - t0

    # 6. Comparative Image Metrics
    min_lines = min(img_raw_20k.shape[0], img_clean.shape[0])
    crop_raw = img_raw_20k[:min_lines]
    crop_clean = img_clean[:min_lines]

    psnr = compute_image_psnr(crop_raw, crop_clean)
    ssim = compute_image_ssim(crop_raw, crop_clean)
    mae = float(np.mean(np.abs(crop_raw - crop_clean)))

    print(f"Results for {pass_name}:")
    print(f"  Input Audio    : {orig_sr} Hz, {dur_sec:.2f} s, peak={peak_amp:.3f}, rms={rms_amp:.3f}")
    print(f"  Denoised Audio : {clean_sr} Hz, {len(clean_data)/clean_sr:.2f} s, peak={float(np.max(np.abs(clean_data))):.3f}")
    print(f"  Denoise Speed  : {t_denoise:.4f} s ({denoise_stats.get('throughput_realtime_x', 0)}x real-time on {denoise_stats.get('device')})")
    print(f"  Decode Latency : Raw native: {t_dec_native*1000:.1f}ms | Raw 20k: {t_dec_raw_20k*1000:.1f}ms | Cleaned: {t_dec_clean*1000:.1f}ms")
    print(f"  Lines Decoded  : Raw native: {meta_native['num_lines']} | Raw 20k: {meta_raw_20k['num_lines']} | Cleaned: {meta_clean['num_lines']}")
    print(f"  Sync Lock %    : Raw native: {meta_native['sync_lock_percentage']}% | Raw 20k: {meta_raw_20k['sync_lock_percentage']}% | Cleaned: {meta_clean['sync_lock_percentage']}%")
    print(f"  Avg Sync Corr  : Raw native: {meta_native['avg_sync_correlation']:.4f} | Raw 20k: {meta_raw_20k['avg_sync_correlation']:.4f} | Cleaned: {meta_clean['avg_sync_correlation']:.4f}")
    print(f"  Image PSNR     : {psnr:.2f} dB")
    print(f"  Image SSIM     : {ssim:.4f}")
    print(f"  Image MAE      : {mae:.4f}")

    # Generate visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{pass_name} — APT Decoder Empirical Comparison\n"
        f"Input: {wav_path.name} ({orig_sr} Hz) -> Production DnCNN-1D -> Cleaned (20,800 Hz)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    t1 = f"1. Raw Audio (Native {orig_sr} Hz) | Lines: {meta_native['num_lines']} | Sync Lock: {meta_native['sync_lock_percentage']}% | Avg Corr: {meta_native['avg_sync_correlation']:.3f}"
    axes[0].imshow(img_native, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title(t1, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Scanlines")

    t2 = f"2. Raw Audio Resampled (20.8 kHz) | Lines: {meta_raw_20k['num_lines']} | Sync Lock: {meta_raw_20k['sync_lock_percentage']}% | Avg Corr: {meta_raw_20k['avg_sync_correlation']:.3f}"
    axes[1].imshow(img_raw_20k, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title(t2, fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Scanlines")

    t3 = f"3. DnCNN-1D Cleaned Audio (20.8 kHz) | Lines: {meta_clean['num_lines']} | Sync Lock: {meta_clean['sync_lock_percentage']}% | Avg Corr: {meta_clean['avg_sync_correlation']:.3f} | PSNR: {psnr:.2f} dB"
    axes[2].imshow(img_clean, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title(t3, fontsize=9, fontweight="bold")
    axes[2].set_ylabel("Scanlines")
    axes[2].set_xlabel("Pixels Across Scanline (0–1039: Channel A Visible | 1040–2079: Channel B Thermal IR)")

    plt.tight_layout()
    fig_path = out_dir / f"{out_prefix}_decoding_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure   : {fig_path}")

    return {
        "pass_name": pass_name,
        "input_wav": str(wav_path),
        "orig_sr": orig_sr,
        "duration_sec": dur_sec,
        "clean_sr": clean_sr,
        "denoise_latency_sec": t_denoise,
        "throughput_realtime_x": denoise_stats.get("throughput_realtime_x"),
        "meta_native": meta_native,
        "meta_raw_20k": meta_raw_20k,
        "meta_clean": meta_clean,
        "psnr_raw_vs_clean": psnr,
        "ssim_raw_vs_clean": ssim,
        "mae_raw_vs_clean": mae,
        "fig_path": str(fig_path),
    }


def main():
    pass1_path = Path("data/raw/audio_137620000Hz_18-06-06_24-01-2025.wav")
    pass2_path = Path("data/raw/audio_137100000Hz_09-19-25_10-12-2024.wav")

    r1 = audit_pass(
        pass_name="NOAA-15 Weak Historical Pass (44.1 kHz, 16.8s)",
        wav_path=pass1_path,
        max_frames=None,
        out_prefix="noaa15_weak_pass",
    )

    r2 = audit_pass(
        pass_name="NOAA-19 High-Elevation Pass (48.0 kHz, 10s slice)",
        wav_path=pass2_path,
        max_frames=int(10.0 * 48000),
        out_prefix="noaa19_zenith_pass",
    )

    # Save combined audit report
    summary = {"noaa15": r1, "noaa19": r2}
    with open("outputs/phase2/multi_pass_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved multi_pass_audit_summary.json successfully!")


if __name__ == "__main__":
    main()
