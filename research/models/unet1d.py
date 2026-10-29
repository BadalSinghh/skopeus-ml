"""1D U-Net Denoiser for Satellite Signal Denoising.

Multi-scale hierarchical encoder-decoder architecture with skip connections,
adapted for raw 1D satellite baseband signals (e.g. NOAA APT downlinks).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Two successive Conv1d layers with normalization and non-linear activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        norm_type: str = "group",
    ):
        super().__init__()
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

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UNet1D(nn.Module):
    """Hierarchical multi-scale 1D U-Net for time-domain signal denoising.

    Features:
      - Multi-scale temporal hierarchy with skip connections to avoid phase smearing.
      - Linear interpolation or transposed convolution upsampling to prevent audio checkerboard artifacts.
      - Optional global residual learning: clean_hat = noisy - predicted_noise.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        channel_mults: List[int] = (1, 2, 4, 8),
        kernel_size: int = 7,
        norm_type: str = "group",
        upsample_mode: str = "linear",
        residual_learning: bool = True,
    ):
        """Initialize 1D U-Net.

        Args:
            in_channels: Number of input signal channels (1 for mono).
            out_channels: Number of output signal channels (1 for mono).
            base_channels: Channel width of the first encoder stage (default: 16).
            channel_mults: Multipliers for each downsampling depth (e.g. [1, 2, 4, 8]).
            kernel_size: Convolution kernel size (odd integer, default: 7).
            norm_type: 'group', 'batch', or 'none'.
            upsample_mode: 'linear' (linear interpolation + conv) or 'conv_transpose'.
            residual_learning: If True, model estimates noise residual and subtracts from input.
        """
        super().__init__()
        self.residual_learning = residual_learning
        self.upsample_mode = upsample_mode
        self.num_stages = len(channel_mults)

        channels = [base_channels * m for m in channel_mults]

        # Encoder stages
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        prev_c = in_channels
        for c in channels:
            self.enc_blocks.append(Conv1DBlock(prev_c, c, kernel_size=kernel_size, norm_type=norm_type))
            self.downsamplers.append(nn.Conv1d(c, c, kernel_size=4, stride=2, padding=1))
            prev_c = c

        # Bottleneck stage
        bottleneck_c = channels[-1] * 2
        self.bottleneck = nn.Sequential(
            Conv1DBlock(channels[-1], bottleneck_c, kernel_size=kernel_size, dilation=1, norm_type=norm_type),
            Conv1DBlock(bottleneck_c, bottleneck_c, kernel_size=kernel_size, dilation=2, norm_type=norm_type),
        )

        # Decoder stages (in reverse order)
        self.upsamplers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        prev_c = bottleneck_c
        for c in reversed(channels):
            if upsample_mode == "conv_transpose":
                self.upsamplers.append(nn.ConvTranspose1d(prev_c, c, kernel_size=4, stride=2, padding=1))
            else:
                self.upsamplers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                        nn.Conv1d(prev_c, c, kernel_size=3, padding=1, bias=False),
                    )
                )
            # Input to dec_block is upsampled features (c) + skip connection (c)
            self.dec_blocks.append(Conv1DBlock(c * 2, c, kernel_size=kernel_size, norm_type=norm_type))
            prev_c = c

        # Final projection head
        self.output_conv = nn.Sequential(
            nn.Conv1d(channels[0], base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy: Tensor of shape [Batch, 1, Length].

        Returns:
            denoised: Tensor of shape [Batch, 1, Length].
        """
        x = noisy
        skips = []

        # Encoder path
        for enc, down in zip(self.enc_blocks, self.downsamplers):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for up, dec, skip in zip(self.upsamplers, self.dec_blocks, reversed(skips)):
            x = up(x)
            # Ensure exact length match between upsampled features and skip connection
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (diff // 2, diff - diff // 2))
            elif diff < 0:
                x = x[:, :, :skip.size(-1)]

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Output projection
        out = self.output_conv(x)

        # Ensure exact length match with input
        diff = noisy.size(-1) - out.size(-1)
        if diff > 0:
            out = F.pad(out, (diff // 2, diff - diff // 2))
        elif diff < 0:
            out = out[:, :, :noisy.size(-1)]

        if self.residual_learning:
            denoised = noisy - out
        else:
            denoised = out

        return denoised

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
