"""Production Standalone Inference Pipeline for 1D DnCNN Satellite Audio Denoising.

Designed for direct integration into web applications, SDR ground stations,
and automated satellite telemetry decoders.

Usage via CLI:
    python scripts/inference_dncnn.py --input noisy.wav --output cleaned.wav

Usage via Python API:
    from scripts.inference_dncnn import denoise_wav

    stats = denoise_wav(
        input_path="noisy.wav",
        output_path="cleaned.wav",
        device="auto",
    )
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import soundfile as sf
import torch

from src.models.dncnn1d import DnCNN1DDenoiser
from src.data.preprocessing import preprocess_signal, resample_audio
from src.data.chunking import slice_into_windows, reconstruct_from_windows


def load_production_model(
    checkpoint_path: Union[str, Path] = "models/dncnn1d_production.pt",
    device: str = "auto",
) -> Tuple[DnCNN1DDenoiser, torch.device, Dict[str, Any]]:
    """Load the trained DnCNN-1D production model onto target device.

    Args:
        checkpoint_path: Path to .pt production checkpoint.
        device: 'auto', 'cuda', 'cuda:0', or 'cpu'.

    Returns:
        (model, selected_device, checkpoint_metadata)
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        # Fallback to research checkpoint if production not found
        fallback = Path("outputs/checkpoints/dncnn1d_baseline_best.pt")
        if fallback.exists():
            ckpt_p = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {fallback}")

    # Determine execution device
    dev_str = device.lower().strip()
    if dev_str == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was explicitly requested ({device}), but no CUDA-enabled GPU is available.")
        target_device = torch.device(device)
    elif dev_str == "cpu":
        target_device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device specification: {device}")

    ckpt_data = torch.load(ckpt_p, map_location="cpu")

    # Extract architecture parameters
    arch_cfg = ckpt_data.get("architecture", {})
    in_channels = arch_cfg.get("in_channels", 1)
    hidden_channels = arch_cfg.get("hidden_channels", 64)
    num_layers = arch_cfg.get("num_layers", 17)
    kernel_size = arch_cfg.get("kernel_size", 5)
    norm_type = arch_cfg.get("norm_type", "batch")
    residual_learning = arch_cfg.get("residual_learning", True)

    model = DnCNN1DDenoiser(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        norm_type=norm_type,
        residual_learning=residual_learning,
    )

    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    return model, target_device, ckpt_data


def denoise_audio_array(
    audio: np.ndarray,
    orig_sr: int,
    model: DnCNN1DDenoiser,
    device: torch.device,
    target_sr: int = 20800,
    window_size: int = 4096,
    hop_size: int = 2048,
    batch_size: int = 32,
    headroom_scale: float = 0.85,
    dc_cutoff_hz: float = 100.0,
    preserve_sample_rate: bool = False,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Denoise a 1D or 2D audio array through the complete DnCNN-1D pipeline.

    Args:
        audio: Input numpy array (float or int PCM).
        orig_sr: Input sample rate in Hz.
        model: Loaded DnCNN1DDenoiser instance.
        device: torch.device to run tensor operations on.
        target_sr: Standardized model sample rate (default: 20800 Hz).
        window_size: Slicing window size in samples.
        hop_size: Hop size for overlap-add.
        batch_size: Batch size for neural inference.
        headroom_scale: Amplitude scaling factor for safe headroom.
        dc_cutoff_hz: Highpass filter cutoff frequency in Hz.
        preserve_sample_rate: If True, resamples output back to orig_sr.

    Returns:
        (cleaned_audio_array, output_sample_rate, metadata_dict)
    """
    t_start = time.perf_counter()

    # Ensure float32 representation
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio_float = (audio.astype(np.float32) / max_val)
    else:
        audio_float = audio.astype(np.float32)

    # 1. Standardized Preprocessing
    prep_audio, prep_meta = preprocess_signal(
        audio_float,
        orig_sr=orig_sr,
        target_sr=target_sr,
        to_mono=True,
        remove_dc=True,
        dc_cutoff_hz=dc_cutoff_hz,
        headroom_scale=headroom_scale,
    )
    scale_factor = prep_meta["scale_factor"]
    num_prep_samples = len(prep_audio)

    if num_prep_samples == 0:
        return np.empty(0, dtype=np.float32), target_sr, {"error": "Empty audio input"}

    # 2. Window Slicing with 50% Overlap
    windows, _ = slice_into_windows(
        prep_audio,
        window_size=window_size,
        hop_size=hop_size,
        pad=True,
    )
    num_windows = len(windows)

    # 3. Batched Deep Learning Forward Passes
    denoised_windows = np.empty_like(windows)
    for start_idx in range(0, num_windows, batch_size):
        end_idx = min(start_idx + batch_size, num_windows)
        batch_arr = windows[start_idx:end_idx]

        # Tensor shape: [Batch, 1, Length]
        batch_tensor = torch.from_numpy(batch_arr).unsqueeze(1).to(device)
        with torch.no_grad():
            out_tensor = model(batch_tensor)
        denoised_windows[start_idx:end_idx] = out_tensor.cpu().numpy().squeeze(1)

    # 4. Overlap-Add (OLA) Synthesis Reconstruction
    denoised_prep = reconstruct_from_windows(
        denoised_windows,
        hop_size=hop_size,
        window_type="hann",
        target_length=num_prep_samples,
    )

    # 5. Invert Headroom Scaling & Clip Protection
    if scale_factor > 1e-12:
        restored = denoised_prep / scale_factor
    else:
        restored = denoised_prep
    cleaned_clipped = np.clip(restored, -1.0, 1.0).astype(np.float32)

    # 6. Sample Rate Preservation (Optional)
    if preserve_sample_rate and orig_sr != target_sr:
        final_audio = resample_audio(cleaned_clipped, orig_sr=target_sr, target_sr=orig_sr)
        final_sr = orig_sr
    else:
        final_audio = cleaned_clipped
        final_sr = target_sr

    t_end = time.perf_counter()
    proc_time_sec = t_end - t_start
    audio_duration_sec = len(audio_float) / float(orig_sr)
    rtf = proc_time_sec / max(audio_duration_sec, 1e-6)

    stats = {
        "input_duration_sec": round(audio_duration_sec, 3),
        "input_sample_rate": orig_sr,
        "output_sample_rate": final_sr,
        "output_duration_sec": round(len(final_audio) / float(final_sr), 3),
        "num_windows_processed": num_windows,
        "processing_time_sec": round(proc_time_sec, 4),
        "real_time_factor": round(rtf, 5),
        "throughput_realtime_x": round(1.0 / max(rtf, 1e-6), 1),
        "peak_input_amplitude": float(np.max(np.abs(audio_float))),
        "peak_output_amplitude": float(np.max(np.abs(final_audio))),
        "scale_factor_applied": float(scale_factor),
        "device_used": str(device),
    }

    return final_audio, final_sr, stats


def denoise_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    checkpoint_path: Union[str, Path] = "models/dncnn1d_production.pt",
    device: str = "auto",
    batch_size: int = 32,
    preserve_sample_rate: bool = False,
    output_subtype: str = "PCM_16",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Production Python API to denoise a satellite audio WAV recording.

    Args:
        input_path: Path to input noisy .wav file.
        output_path: Destination path for cleaned output .wav file.
        checkpoint_path: Path to dncnn1d_production.pt checkpoint.
        device: 'auto', 'cuda', or 'cpu'.
        batch_size: Mini-batch size for windowed inference.
        preserve_sample_rate: If True, output has input's native sample rate.
                             If False (default), output is standardized to 20800 Hz.
        output_subtype: Audio encoding ('PCM_16' or 'FLOAT').
        verbose: If True, prints formatted summary to stdout.

    Returns:
        Dictionary of execution metrics and signal properties.
    """
    inp_p = Path(input_path)
    out_p = Path(output_path)
    if not inp_p.exists():
        raise FileNotFoundError(f"Input audio file not found: {inp_p}")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load production model
    model, target_device, meta = load_production_model(checkpoint_path=checkpoint_path, device=device)

    # 2. Read input audio
    data, sample_rate = sf.read(str(inp_p), dtype="float32")
    channels = 1 if data.ndim == 1 else data.shape[1]

    # 3. Execute inference pipeline
    cleaned, final_sr, stats = denoise_audio_array(
        audio=data,
        orig_sr=sample_rate,
        model=model,
        device=target_device,
        target_sr=20800,
        batch_size=batch_size,
        preserve_sample_rate=preserve_sample_rate,
    )

    # 4. Save cleaned WAV file
    sf.write(str(out_p), cleaned, final_sr, subtype=output_subtype)

    # Get device hardware name
    gpu_name = "N/A (CPU execution)"
    if target_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(target_device)

    full_stats = {
        "model_name": meta.get("model_name", "DnCNN-1D (17L)"),
        "checkpoint_version": meta.get("version", "1.0.0"),
        "checkpoint_path": str(checkpoint_path),
        "device": str(target_device).upper(),
        "gpu_name": gpu_name,
        "input_path": str(inp_p),
        "output_path": str(out_p),
        "input_channels": channels,
        "output_channels": 1,
        "output_subtype": output_subtype,
        **stats,
    }

    if verbose:
        print("\n" + "=" * 65)
        print("DnCNN-1D SATELLITE AUDIO DENOISING INFERENCE PIPELINE")
        print("=" * 65)
        print(f"Model          : {full_stats['model_name']} (v{full_stats['checkpoint_version']})")
        print(f"Device         : {full_stats['device']} ({gpu_name})")
        print(f"Input File     : {inp_p.name} ({full_stats['input_duration_sec']} s, {sample_rate} Hz, {channels} ch)")
        print(f"Output File    : {out_p.name} ({full_stats['output_duration_sec']} s, {final_sr} Hz, mono, {output_subtype})")
        print(f"Windows Processed: {full_stats['num_windows_processed']} (4096 samples, 50% OLA)")
        print(f"Processing Time: {full_stats['processing_time_sec']:.4f} s ({full_stats['throughput_realtime_x']}x faster than real-time)")
        print(f"Peak Amplitude : Input {full_stats['peak_input_amplitude']:.3f} -> Output {full_stats['peak_output_amplitude']:.3f}")
        print("=" * 65 + "\n")

    return full_stats


def main():
    parser = argparse.ArgumentParser(
        description="Denoise satellite audio recordings using production DnCNN-1D"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input noisy .wav file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to cleaned output .wav file")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default="models/dncnn1d_production.pt",
        help="Path to production .pt checkpoint (default: models/dncnn1d_production.pt)"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Execution device: 'auto' (GPU if available), 'cuda', or 'cpu'"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size for windowed inference (default: 32)"
    )
    parser.add_argument(
        "--preserve-sr", action="store_true",
        help="Resample output back to original input sample rate instead of standardized 20800 Hz"
    )
    parser.add_argument(
        "--format", choices=["PCM_16", "FLOAT"], default="PCM_16",
        help="WAV sample format (default: PCM_16)"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress console summary")

    args = parser.parse_args()

    try:
        denoise_wav(
            input_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            preserve_sample_rate=args.preserve_sr,
            output_subtype=args.format,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error during denoising inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
