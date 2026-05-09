"""
FSPPF — Feature-fusion Spatial Pyramid Pooling Fast for FD-YOLO11
==================================================================
Enhanced spatial pyramid pooling that fuses local and global information
more effectively than standard SPPF. Introduces residual connections
and multi-branch feature aggregation for improved defect recognition
in complex industrial backgrounds.

Key Innovation:
- Multi-scale pooling with learned residual fusion
- Cross-scale feature interaction via 1x1 convolutions
- Better local-global context aggregation than standard SPPF

Reference: FD-YOLO11 (IEEE Access, 2025) — Section III-B
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FSPPF(nn.Module):
    """Feature-fusion Spatial Pyramid Pooling Fast.

    Extends the standard SPPF by adding learned residual connections
    and multi-scale feature fusion branches. This enables better
    contextual understanding for detecting defects that vary in size
    (from tiny scratches to large surface patches).

    Architecture:
        Input → Conv1x1 (channel reduction)
            ↓
        ┌── MaxPool(k=5) ──── MaxPool(k=5) ──── MaxPool(k=5) ──┐
        │        ↓                   ↓                ↓          │
        │    [pool_1]            [pool_2]          [pool_3]      │
        │        ↓                   ↓                ↓          │
        └── Concat(input, pool_1, pool_2, pool_3) ──────────────┘
            ↓
        Fusion Conv (1x1 channel mixing) + Residual
            ↓
        Output Conv1x1 (channel projection)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Pooling kernel size (default: 5).
        expansion: Channel expansion ratio for intermediate features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()

        hidden = int(in_channels * expansion)

        # Input channel reduction
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        # Sequential max-pooling with same padding
        padding = kernel_size // 2
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        # Feature fusion: 4 * hidden channels (input + 3 pooling levels)
        # Cross-scale interaction convolution
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 4, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(hidden * 4),
            nn.SiLU(inplace=True),
        )

        # Channel mixing after fusion — enables cross-scale information flow
        self.conv_mix = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden * 4),
            nn.SiLU(inplace=True),
        )

        # Output projection
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Residual connection (match dimensions if needed)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale pooling and residual fusion.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Feature-fused tensor of shape (B, C_out, H, W).
        """
        residual = self.residual(x)

        # Channel reduction
        y = self.conv_in(x)

        # Multi-scale spatial pyramid pooling
        p1 = self.pool(y)
        p2 = self.pool(p1)
        p3 = self.pool(p2)

        # Concatenate all scales: [original, 1x pool, 2x pool, 3x pool]
        concat = torch.cat([y, p1, p2, p3], dim=1)

        # Feature fusion with cross-scale mixing
        fused = self.conv_fuse(concat)
        mixed = self.conv_mix(fused)

        # Residual connection for gradient flow
        fused_final = fused + mixed  # Self-residual within fusion

        # Project to output channels + input residual
        out = self.conv_out(fused_final)
        return out + residual
