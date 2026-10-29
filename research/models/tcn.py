"""Temporal Convolutional Network (TCN) for 1D Satellite Signal Denoising.

Employs dilated non-causal convolutional residual blocks with exponentially increasing
dilation factors to provide large temporal receptive fields for modeling satellite
carrier periodicity and wideband noise.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Residual block composed of two dilated Conv1d layers with normalization and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.05,
        norm_type: str = "group",
    ):
        super().__init__()
        # Symmetric padding for non-causal processing (reconstructs full window)
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        num_groups = min(4, out_channels)
        if norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        elif norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # 1x1 projection for residual match if channels change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)

        out = self.act1(self.norm1(self.conv1(x)))
        out = self.drop1(out)

        out = self.norm2(self.conv2(out))
        out = self.drop2(out)

        # Ensure spatial dimension match
        if out.shape[-1] != res.shape[-1]:
            diff = res.shape[-1] - out.shape[-1]
            if diff > 0:
                out = nn.functional.pad(out, (diff // 2, diff - diff // 2))
            else:
                out = out[:, :, :res.shape[-1]]

        return self.act2(out + res)


class TCNDenoiser(nn.Module):
    """Dilated Temporal Convolutional Network for raw time-domain signal denoising."""

    def __init__(
        self,
        in_channels: int = 1,
        num_channels: List[int] = (64, 64, 64, 64, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.05,
        norm_type: str = "group",
        residual_learning: bool = True,
    ):
        """Initialize TCN.

        Args:
            in_channels: Number of input signal channels (1 for mono).
            num_channels: Hidden channel width for each dilated stage.
            kernel_size: Convolution kernel size (odd integer, default: 5).
            dropout: Dropout probability.
            norm_type: 'group' or 'batch'.
            residual_learning: If True, model estimates noise residual and subtracts from input.
        """
        super().__init__()
        self.residual_learning = residual_learning
        self.receptive_field = 1

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )
            self.receptive_field += 2 * (kernel_size - 1) * dilation

        self.network = nn.Sequential(*layers)

        # Output projection head (maps back to 1 channel)
        self.output_head = nn.Sequential(
            nn.Conv1d(num_channels[-1], num_channels[-1] // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(num_channels[-1] // 2, in_channels, kernel_size=1),
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy: Tensor of shape [Batch, 1, Length].

        Returns:
            denoised: Tensor of shape [Batch, 1, Length].
        """
        features = self.network(noisy)
        noise_pred = self.output_head(features)

        if self.residual_learning:
            denoised = noisy - noise_pred
        else:
            denoised = noise_pred

        return denoised

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
