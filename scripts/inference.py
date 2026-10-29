"""Standalone CLI Inference Tool for Skopeus ML DnCNN-1D.

Allows local testing without launching the FastAPI web server.
Usage:
    python scripts/inference.py input.wav output_cleaned.wav [--device cuda|cpu]
"""

import argparse
import sys
from pathlib import Path

# Add root directory to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from denoise_service import denoise_wav


def main():
    parser = argparse.ArgumentParser(description="DnCNN-1D Satellite Signal Denoising CLI")
    parser.add_argument("input_wav", type=str, help="Path to input noisy WAV recording")
    parser.add_argument("output_wav", type=str, help="Path to output denoised WAV file")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")

    args = parser.parse_args()

    print(f"Loading certified DnCNN-1D model...")
    stats = denoise_wav(args.input_wav, args.output_wav, device=args.device)

    print("\n--- Processing Summary ---")
    print(f"Input:         {stats['input_file']} ({stats['input_sample_rate']} Hz, {stats['input_duration_sec']}s)")
    print(f"Output:        {stats['output_file']} ({stats['output_sample_rate']} Hz, mono PCM 16)")
    print(f"Device:        {stats['device']}")
    print(f"Time:          {stats['processing_time_sec']}s ({stats['throughput_realtime_x']}x realtime)")
    print(f"SHA-256:       {stats['checkpoint_sha256'][:16]}... (Certified)")
    print("Done!")


if __name__ == "__main__":
    main()
