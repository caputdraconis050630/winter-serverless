"""Shared body architectures: TCN, GRU, PatchTST.

Each body maps input [B, L, F] -> embedding [B, d].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal convolution with dilation."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Residual TCN block with two causal convolutions."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [B, T, C]
        out = self.norm1(out)
        out = out.transpose(1, 2)  # [B, C, T]
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + residual


class TCNBody(nn.Module):
    """Temporal Convolutional Network body.

    Input: [B, L, F] -> Output: [B, d]
    """
    def __init__(self, in_features, embedding_dim=64, n_blocks=5,
                 channels=64, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, channels)
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, dilation=2**i)
            for i in range(n_blocks)
        ])
        self.output_proj = nn.Linear(channels, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: [B, L, F]
        x = self.input_proj(x)       # [B, L, C]
        x = x.transpose(1, 2)         # [B, C, L]
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)         # [B, L, C]
        # Take last timestep
        x = x[:, -1, :]               # [B, C]
        x = self.output_proj(x)        # [B, d]
        return x


class GRUBody(nn.Module):
    """GRU body.

    Input: [B, L, F] -> Output: [B, d]
    """
    def __init__(self, in_features, embedding_dim=64, n_layers=2,
                 hidden_size=128):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_size, num_layers=n_layers,
                          batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: [B, L, F]
        out, _ = self.gru(x)          # [B, L, H]
        out = out[:, -1, :]            # [B, H] last timestep
        out = self.output_proj(out)    # [B, d]
        return out


class PatchTSTBody(nn.Module):
    """Simplified PatchTST body.

    Patches the input sequence, applies Transformer encoder, mean-pools.
    Input: [B, L, F] -> Output: [B, d]
    """
    def __init__(self, in_features, embedding_dim=64, patch_size=8,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # Patch embedding
        self.patch_proj = nn.Linear(patch_size * in_features, d_model)

        # Positional encoding
        max_patches = 256
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, embedding_dim)

    def forward(self, x):
        # x: [B, L, F]
        B, L, F = x.shape
        n_patches = L // self.patch_size
        # Trim to exact patches
        x = x[:, :n_patches * self.patch_size, :]
        # Reshape to patches: [B, n_patches, patch_size * F]
        x = x.reshape(B, n_patches, self.patch_size * F)
        # Project
        x = self.patch_proj(x)         # [B, n_patches, d_model]
        # Add positional encoding
        x = x + self.pos_embed[:, :n_patches, :]
        # Transformer
        x = self.transformer(x)        # [B, n_patches, d_model]
        # Mean pool
        x = x.mean(dim=1)             # [B, d_model]
        # Project to embedding
        x = self.output_proj(x)        # [B, d]
        return x


def build_body(body_type, in_features, cfg=None):
    """Factory for body architectures."""
    embedding_dim = 64 if cfg is None else cfg.get("embedding_dim", 64)

    if body_type == "tcn":
        return TCNBody(
            in_features, embedding_dim,
            n_blocks=5 if cfg is None else cfg.get("tcn", {}).get("n_blocks", 5),
            channels=64 if cfg is None else cfg.get("tcn", {}).get("channels", 64),
            kernel_size=3 if cfg is None else cfg.get("tcn", {}).get("kernel_size", 3),
        )
    elif body_type == "gru":
        return GRUBody(
            in_features, embedding_dim,
            n_layers=2 if cfg is None else cfg.get("gru", {}).get("n_layers", 2),
            hidden_size=128 if cfg is None else cfg.get("gru", {}).get("hidden_size", 128),
        )
    elif body_type == "patchtst":
        return PatchTSTBody(
            in_features, embedding_dim,
            patch_size=8 if cfg is None else cfg.get("patchtst", {}).get("patch_size", 8),
            d_model=128 if cfg is None else cfg.get("patchtst", {}).get("d_model", 128),
            n_heads=4 if cfg is None else cfg.get("patchtst", {}).get("n_heads", 4),
            n_layers=3 if cfg is None else cfg.get("patchtst", {}).get("n_layers", 3),
        )
    else:
        raise ValueError(f"Unknown body type: {body_type}")
