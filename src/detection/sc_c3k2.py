"""
SC-C3k2 — Self-Calibrated C3k2 Module for FD-YOLO11
=====================================================
Replaces the standard C3k2 block in YOLO11 with self-calibrated convolutions
that adaptively capture contextual information across multiple receptive fields.

Key Innovation:
- Self-calibration branch re-weights feature channels using learned attention
- Multi-receptive-field fusion improves detection of defects at varying scales
- Maintains computational efficiency comparable to baseline C3k2

Reference: FD-YOLO11 (IEEE Access, 2025) — Section III-A
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCalibratedConv(nn.Module):
    """Self-Calibrated Convolution block.

    Implements a two-branch architecture:
    1. Standard convolution branch for spatial feature extraction
    2. Self-calibration branch that generates channel attention weights
       from downsampled features, enabling adaptive feature re-weighting.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding (auto-computed if None).
        groups: Number of groups for grouped convolution.
        pool_size: Pooling size for the self-calibration branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        pool_size: int = 4,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # ── Standard convolution branch ──
        self.conv_standard = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # ── Self-calibration branch ──
        # Downsampled pathway for global context
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv_calibrate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Attention generator: produces per-channel calibration weights
        self.conv_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

        # Fusion projection after calibration
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-calibrated attention.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Calibrated feature tensor of shape (B, C_out, H, W).
        """
        # Standard convolution path
        feat_standard = self.conv_standard(x)

        # Self-calibration path
        _, _, h, w = feat_standard.shape
        pooled = self.pool(x)  # (B, C_in, pool_size, pool_size)
        calibrated = self.conv_calibrate(pooled)  # (B, C_out, pool_size, pool_size)

        # Upsample calibration to match spatial dimensions
        calibrated = F.interpolate(calibrated, size=(h, w), mode="bilinear", align_corners=False)

        # Generate attention weights
        attention = self.conv_attention(calibrated)  # (B, C_out, H, W), values in [0, 1]

        # Apply calibration: re-weight standard features
        calibrated_feat = feat_standard * attention + feat_standard

        # Final fusion
        return self.conv_fuse(calibrated_feat)


class Bottleneck_SCC(nn.Module):
    """Bottleneck block with Self-Calibrated Convolution.

    Replaces the standard 3x3 conv in the bottleneck with SelfCalibratedConv
    for enhanced multi-scale feature extraction.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        shortcut: Whether to use residual connection.
        groups: Groups for grouped convolution.
        expansion: Channel expansion ratio for the bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()

        hidden = int(out_channels * expansion)
        self.use_shortcut = shortcut and in_channels == out_channels

        # 1x1 channel reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        # 3x3 self-calibrated convolution (core enhancement)
        self.conv2 = SelfCalibratedConv(hidden, out_channels, kernel_size=3, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        out = self.conv2(self.conv1(x))
        return out + x if self.use_shortcut else out


class SCC3k2(nn.Module):
    """SC-C3k2 — Self-Calibrated C3k2 Module.

    Drop-in replacement for YOLO11's C3k2 block. Uses split-and-concat
    architecture with Self-Calibrated Bottleneck blocks for improved
    multi-scale defect feature extraction.

    Architecture:
        Input → Conv1x1 → Split into 2 parts
        Part 1: Direct passthrough
        Part 2: N × Bottleneck_SCC blocks (sequential)
        Concat all parts → Conv1x1 → Output

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        n_blocks: Number of bottleneck blocks.
        shortcut: Whether to use residual connections in bottlenecks.
        groups: Groups for grouped convolution.
        expansion: Channel expansion ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()

        hidden = int(out_channels * expansion)

        # Input projection: map to 2 * hidden channels, then split
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 2 * hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * hidden),
            nn.SiLU(inplace=True),
        )

        # Self-calibrated bottleneck blocks
        self.blocks = nn.ModuleList([
            Bottleneck_SCC(hidden, hidden, shortcut=shortcut, groups=groups, expansion=1.0)
            for _ in range(n_blocks)
        ])

        # Output projection: concat all splits → out_channels
        # Total concat channels = hidden (pass-through) + hidden * n_blocks (from each block)
        concat_channels = hidden * (1 + n_blocks)
        self.conv_out = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: split → bottleneck chain → concat → project.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        # Project and split into pass-through + processing
        projected = self.conv_in(x)
        parts = projected.chunk(2, dim=1)
        y_passthrough = parts[0]
        y_process = parts[1]

        # Collect outputs: passthrough + each bottleneck's output
        outputs = [y_passthrough]
        for block in self.blocks:
            y_process = block(y_process)
            outputs.append(y_process)

        # Concat all and project to output channels
        return self.conv_out(torch.cat(outputs, dim=1))
