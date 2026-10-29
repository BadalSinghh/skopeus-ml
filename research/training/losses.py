"""Configurable time-domain and frequency-domain loss functions for signal denoising."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


class STFTLoss(nn.Module):
    """Single-resolution STFT magnitude and spectral convergence loss."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 1, L] -> flatten to [B, L]
        p = pred.squeeze(1)
        t = target.squeeze(1)

        # Compute STFT
        window = self.window.to(pred.device)
        P = torch.stft(p, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, return_complex=True)
        T = torch.stft(t, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, return_complex=True)

        mag_p = torch.abs(P)
        mag_t = torch.abs(T)

        # Spectral convergence loss
        sc_loss = torch.norm(mag_t - mag_p, p="fro") / (torch.norm(mag_t, p="fro") + 1e-8)

        # Log magnitude STFT loss
        log_mag_p = torch.log(mag_p + 1e-7)
        log_mag_t = torch.log(mag_t + 1e-7)
        log_loss = nn.functional.l1_loss(log_mag_p, log_mag_t)

        return sc_loss + log_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss across varying temporal and spectral window lengths."""

    def __init__(
        self,
        fft_sizes: List[int] = (256, 512, 1024),
        hop_sizes: List[int] = (64, 128, 256),
        win_lengths: List[int] = (256, 512, 1024),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.resolutions = nn.ModuleList([
            STFTLoss(n_fft=f, hop_length=h, win_length=w)
            for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for stft_layer in self.resolutions:
            loss = loss + stft_layer(pred, target)
        return loss / len(self.resolutions)


class CombinedLoss(nn.Module):
    """Composite loss function: L = lambda_time * L_time + lambda_freq * L_freq."""

    def __init__(
        self,
        time_loss_type: str = "mse",
        lambda_time: float = 1.0,
        lambda_freq: float = 0.5,
    ):
        super().__init__()
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq

        if time_loss_type == "mse":
            self.time_loss = nn.MSELoss()
        elif time_loss_type == "l1":
            self.time_loss = nn.L1Loss()
        elif time_loss_type == "smooth_l1":
            self.time_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown time loss: {time_loss_type}")

        if lambda_freq > 0.0:
            self.freq_loss = MultiResolutionSTFTLoss()
        else:
            self.freq_loss = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.lambda_time * self.time_loss(pred, target)
        if self.freq_loss is not None and self.lambda_freq > 0.0:
            loss = loss + self.lambda_freq * self.freq_loss(pred, target)
        return loss


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build loss function based on YAML configuration."""
    loss_name = config.get("name", "mse").lower()
    lambda_time = config.get("lambda_time", 1.0)
    lambda_freq = config.get("lambda_freq", 0.0)

    if loss_name in ("mse", "l1", "smooth_l1"):
        if lambda_freq > 0.0:
            return CombinedLoss(time_loss_type=loss_name, lambda_time=lambda_time, lambda_freq=lambda_freq)
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "l1":
            return nn.L1Loss()
        elif loss_name == "smooth_l1":
            return nn.SmoothL1Loss()

    elif loss_name in ("spectral", "stft"):
        return MultiResolutionSTFTLoss()

    elif loss_name == "combined":
        time_type = config.get("time_loss", "mse")
        return CombinedLoss(time_loss_type=time_type, lambda_time=lambda_time, lambda_freq=lambda_freq)

    raise ValueError(f"Unsupported loss configuration: {config}")
