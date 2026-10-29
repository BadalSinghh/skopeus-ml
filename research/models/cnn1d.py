"""1D Convolutional Neural Network baseline denoiser for satellite signals."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """Convolutional layer with GroupNorm and LeakyReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        # GroupNorm is stable across arbitrary batch sizes (4, 8, 16) on laptop GPU
        num_groups = min(4, out_channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CNN1DDenoiser(nn.Module):
    """Standard 1D CNN baseline architecture for audio / baseband denoising."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 48,
        num_layers: int = 5,
        kernel_size: int = 7,
        residual_learning: bool = True,
    ):
        """Initialize 1D CNN.

        Args:
            in_channels: Input channels (1 for mono audio).
            hidden_channels: Channel width of intermediate layers.
            num_layers: Total number of convolutional blocks.
            kernel_size: Convolution kernel width.
            residual_learning: If True, model predicts noise residual: clean = noisy - noise.
        """
        super().__init__()
        self.residual_learning = residual_learning

        layers = []
        # Input layer
        layers.append(Conv1DBlock(in_channels, hidden_channels, kernel_size=kernel_size))

        # Intermediate feature extraction layers with mild dilation progression
        for i in range(num_layers - 2):
            dilation = 2 ** (i % 3)
            layers.append(Conv1DBlock(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=dilation))

        # Pre-output block
        layers.append(Conv1DBlock(hidden_channels, hidden_channels, kernel_size=kernel_size))

        self.features = nn.Sequential(*layers)

        # Output projection back to 1 channel (linear, no norm/act)
        self.output_conv = nn.Conv1d(
            hidden_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy: Tensor of shape [Batch, 1, Length].

        Returns:
            denoised: Tensor of shape [Batch, 1, Length].
        """
        feats = self.features(noisy)
        out = self.output_conv(feats)

        if self.residual_learning:
            # Model predicts noise residual: clean = noisy - predicted_noise
            denoised = noisy - out
        else:
            denoised = out

        return denoised

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
