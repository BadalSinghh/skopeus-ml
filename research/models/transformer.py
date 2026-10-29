"""1D Transformer Denoiser for Raw Satellite Baseband Signals.

Utilizes a 1D patch embedding tokenizer, sinusoidal/learnable positional encodings,
multi-head self-attention encoder blocks, and a convolutional reconstruction head
to capture global carrier periodicity, wideband interference, and non-local correlations.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for 1D token sequences."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, d_model]
        return x + self.pe[:, : x.size(1), :]


class Transformer1DDenoiser(nn.Module):
    """1D Transformer denoiser for time-domain satellite baseband signals."""

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.05,
        residual_learning: bool = True,
    ):
        """Initialize 1D Transformer Denoiser.

        Args:
            in_channels: Number of input signal channels (1 for mono).
            patch_size: Temporal patch size P (samples per token, default: 16).
            d_model: Transformer hidden representation dimension (default: 128).
            nhead: Number of attention heads (default: 4).
            num_layers: Number of Transformer encoder layers (default: 4).
            dim_feedforward: Intermediate MLP dimension (default: 256).
            dropout: Dropout probability.
            residual_learning: If True, model estimates noise residual and subtracts from input.
        """
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.residual_learning = residual_learning

        # 1. Patch Tokenizer (1D Convolution)
        self.patch_embed = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.embed_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding1D(d_model=d_model, max_len=4096)

        # 2. Transformer Encoder (Pre-LN formulation for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # 3. Waveform Reconstruction Head
        # Transposed convolution projects tokens back to full temporal sample rate
        self.decoder_transconv = nn.ConvTranspose1d(
            d_model,
            d_model // 2,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Refinement convolution smoothing patch seams and tracking exact carrier phase
        self.refinement = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(d_model // 2, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy: Tensor of shape [Batch, 1, Length].

        Returns:
            denoised: Tensor of shape [Batch, 1, Length].
        """
        orig_len = noisy.shape[-1]

        # Pad to multiple of patch_size if needed
        pad_len = (self.patch_size - (orig_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = F.pad(noisy, (0, pad_len))
        else:
            x = noisy

        # Tokenize: [B, 1, L_padded] -> [B, d_model, T] -> [B, T, d_model]
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = self.embed_norm(tokens)
        tokens = self.pos_encoder(tokens)

        # Transformer self-attention: [B, T, d_model]
        encoded = self.transformer_encoder(tokens)

        # Reshape for reconstruction head: [B, d_model, T]
        encoded_t = encoded.transpose(1, 2)

        # Upsample back to temporal sample level: [B, d_model // 2, L_padded]
        upsampled = self.decoder_transconv(encoded_t)
        noise_pred = self.refinement(upsampled)

        # Trim padding if any was added
        if pad_len > 0:
            noise_pred = noise_pred[:, :, :orig_len]

        if self.residual_learning:
            denoised = noisy - noise_pred
        else:
            denoised = noise_pred

        return denoised

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
