"""1D Deep Convolutional Neural Network (DnCNN) for Satellite Signal Denoising.

Self-contained production architecture definition.
Parameter count: 309,825 parameters (17 layers, 64 hidden channels, kernel size 5).
"""

import torch
import torch.nn as nn


class DnCNN1DDenoiser(nn.Module):
    """1D adaptation of the deep residual DnCNN architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 17,
        kernel_size: int = 5,
        norm_type: str = "batch",
        residual_learning: bool = True,
    ):
        super().__init__()
        assert num_layers >= 3, f"DnCNN requires at least 3 layers, got {num_layers}"
        assert kernel_size % 2 == 1, f"kernel_size must be odd for symmetric padding, got {kernel_size}"

        self.residual_learning = residual_learning
        padding = kernel_size // 2

        layers = []
        # Layer 1: Conv + ReLU (no normalization)
        layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Layers 2 to D-1: Conv + Normalization + ReLU
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False)
            )
            if norm_type.lower() == "batch":
                layers.append(nn.BatchNorm1d(hidden_channels))
            elif norm_type.lower() == "group":
                num_groups = min(4, hidden_channels)
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels))
            elif norm_type.lower() == "none":
                pass
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
            layers.append(nn.ReLU(inplace=True))

        # Layer D: Conv (projection back to in_channels, no norm, no activation)
        layers.append(
            nn.Conv1d(hidden_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=True)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting residual noise and subtracting."""
        noise_residual = self.network(noisy)

        if self.residual_learning:
            denoised = noisy - noise_residual
        else:
            denoised = noise_residual

        return denoised

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
