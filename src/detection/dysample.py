"""
DySample — Dynamic Upsampling for FD-YOLO11
=============================================
Replaces traditional nearest-neighbor / bilinear upsampling with a
learnable, content-aware upsampling mechanism. This reduces semantic
information loss during feature map upscaling in the FPN neck.

Key Innovation:
- Generates dynamic sampling offsets conditioned on input features
- Lightweight: minimal parameter overhead vs. deformable convolutions
- Preserves fine-grained defect boundary information during upsampling

Reference: FD-YOLO11 (IEEE Access, 2025) — Section III-C
Based on: "Learning to Upsample by Learning to Sample" (ICCV 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Dynamic Upsampling Module.

    Instead of using fixed interpolation kernels, DySample generates
    per-pixel sampling locations conditioned on input features. This
    content-aware upsampling preserves spatial details critical for
    detecting fine defects like scratches and crazing.

    The module works by:
    1. Generating a 2D offset field from input features via a lightweight conv
    2. Creating a base sampling grid (identity grid)
    3. Adding the learned offsets to the base grid
    4. Using grid_sample to resample the input at the offset locations

    Args:
        in_channels: Number of input channels.
        scale_factor: Upsampling factor (default: 2).
        style: Offset generation style — 'lp' (linear projection)
               or 'pl' (pixel-shuffle based).
        groups: Groups for offset generation convolution.
    """

    def __init__(
        self,
        in_channels: int,
        scale_factor: int = 2,
        style: str = "lp",
        groups: int = 4,
    ) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.style = style
        self.groups = groups

        if style == "lp":
            # Linear projection style: generate offsets then pixel-shuffle
            self.offset_conv = nn.Sequential(
                nn.Conv2d(in_channels, 2 * groups * scale_factor ** 2, kernel_size=1, bias=False),
                nn.PixelShuffle(scale_factor),
            )
        elif style == "pl":
            # Pixel-shuffle first, then linear projection for offsets
            self.offset_conv = nn.Sequential(
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    in_channels // (scale_factor ** 2),
                    2 * groups,
                    kernel_size=1,
                    bias=False,
                ),
            )
        else:
            raise ValueError(f"Unknown DySample style: {style}. Use 'lp' or 'pl'.")

        # Initialize offset convolution to near-zero for stable training start
        self._init_weights()

        # Cache for the base sampling grid
        self._grid_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _init_weights(self) -> None:
        """Initialize offset weights to near-zero for identity-like start."""
        for m in self.offset_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_base_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve cached normalized base sampling grid.

        Args:
            h: Output height.
            w: Output width.
            device: Target device.

        Returns:
            Base grid of shape (1, h, w, 2) with values in [-1, 1].
        """
        key = (h, w)
        if key not in self._grid_cache or self._grid_cache[key].device != device:
            # Create normalized coordinate grid: [-1, 1] range
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
            self._grid_cache[key] = grid

        return self._grid_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: generate offsets and resample input.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Upsampled tensor of shape (B, C, H*scale, W*scale).
        """
        b, c, h, w = x.shape
        h_out = h * self.scale_factor
        w_out = w * self.scale_factor

        # Generate offset field: (B, 2*groups, H_out, W_out)
        offsets = self.offset_conv(x)

        # Average across groups: (B, 2, H_out, W_out)
        offsets = offsets.reshape(b, 2, self.groups, h_out, w_out).mean(dim=2)

        # Permute to grid_sample format: (B, H_out, W_out, 2)
        offsets = offsets.permute(0, 2, 3, 1).contiguous()

        # Scale offsets to normalized coordinate range
        # Offsets should be small perturbations, normalize by output size
        offset_scale = torch.tensor(
            [2.0 / w_out, 2.0 / h_out],
            device=x.device, dtype=x.dtype,
        )
        offsets = offsets * offset_scale

        # Get base sampling grid and add learned offsets
        base_grid = self._get_base_grid(h_out, w_out, x.device)
        grid = base_grid.expand(b, -1, -1, -1) + offsets

        # Upsample input to target resolution first (for grid_sample input)
        x_up = F.interpolate(x, size=(h_out, w_out), mode="bilinear", align_corners=False)

        # Resample with learned grid
        output = F.grid_sample(
            x_up,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        return output
