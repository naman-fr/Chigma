"""
FD-YOLO11 — Feature-Enhanced YOLO11 for Industrial Defect Detection
=====================================================================
Integrates SC-C3k2, FSPPF, and DySample into YOLO11 architecture.
Reference: IEEE Access, 2025
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from loguru import logger

from src.detection.dysample import DySample
from src.detection.fsppf import FSPPF
from src.detection.sc_c3k2 import SCC3k2


class FDYOLOBackbone(nn.Module):
    """Backbone with SC-C3k2 blocks at each stage."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(True),
            SCC3k2(c2, c2, n_blocks=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(True),
            SCC3k2(c3, c3, n_blocks=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, 2, 1, bias=False), nn.BatchNorm2d(c4), nn.SiLU(True),
            SCC3k2(c4, c4, n_blocks=2),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c4, c5, 3, 2, 1, bias=False), nn.BatchNorm2d(c5), nn.SiLU(True),
            SCC3k2(c5, c5, n_blocks=2),
        )
        self.fsppf = FSPPF(c5, c5)
        self.out_channels = [c3, c4, c5]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.fsppf(self.stage4(p4))
        return p3, p4, p5


class FDYOLONeck(nn.Module):
    """FPN Neck with DySample upsampling."""

    def __init__(self, in_channels: list[int]) -> None:
        super().__init__()
        c3, c4, c5 = in_channels

        # Top-down
        self.dysample_p5 = DySample(c5, scale_factor=2)
        self.reduce_p5 = nn.Sequential(nn.Conv2d(c5, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.SiLU(True))
        self.fuse_p4 = SCC3k2(c4 * 2, c4, n_blocks=2, shortcut=False)

        self.dysample_p4 = DySample(c4, scale_factor=2)
        self.reduce_p4 = nn.Sequential(nn.Conv2d(c4, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(True))
        self.fuse_p3 = SCC3k2(c3 * 2, c3, n_blocks=2, shortcut=False)

        # Bottom-up
        self.down_p3 = nn.Sequential(nn.Conv2d(c3, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(True))
        self.fuse_p4_bu = SCC3k2(c3 + c4, c4, n_blocks=2, shortcut=False)
        self.down_p4 = nn.Sequential(nn.Conv2d(c4, c4, 3, 2, 1, bias=False), nn.BatchNorm2d(c4), nn.SiLU(True))
        self.fuse_p5_bu = SCC3k2(c4 + c5, c5, n_blocks=2, shortcut=False)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        p3, p4, p5 = features
        # Top-down
        p5_up = self.reduce_p5(self.dysample_p5(p5))
        p4_td = self.fuse_p4(torch.cat([p5_up, p4], 1))
        p4_up = self.reduce_p4(self.dysample_p4(p4_td))
        p3_td = self.fuse_p3(torch.cat([p4_up, p3], 1))
        # Bottom-up
        p3_d = self.down_p3(p3_td)
        p4_bu = self.fuse_p4_bu(torch.cat([p3_d, p4_td], 1))
        p4_d = self.down_p4(p4_bu)
        p5_bu = self.fuse_p5_bu(torch.cat([p4_d, p5], 1))
        return p3_td, p4_bu, p5_bu


class DetectionHead(nn.Module):
    """Decoupled detection head with DWConv."""

    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 16) -> None:
        super().__init__()
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.SiLU(True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels), nn.SiLU(True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.SiLU(True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels), nn.SiLU(True),
        )
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_pred(self.cls_conv(x)), self.reg_pred(self.reg_conv(x))


class FDYOLO11(nn.Module):
    """Complete FD-YOLO11 model.

    Args:
        num_classes: Number of defect classes.
        base_channels: Base channel count (scales entire model).
        reg_max: DFL regression range.
    """

    def __init__(self, num_classes: int = 6, base_channels: int = 64, reg_max: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = FDYOLOBackbone(in_channels=3, base_channels=base_channels)
        self.neck = FDYOLONeck(self.backbone.out_channels)
        c3, c4, c5 = self.backbone.out_channels
        self.head_p3 = DetectionHead(c3, num_classes, reg_max)
        self.head_p4 = DetectionHead(c4, num_classes, reg_max)
        self.head_p5 = DetectionHead(c5, num_classes, reg_max)
        self._initialize_weights()
        logger.info(f"FD-YOLO11: {num_classes} classes, {self.count_parameters()/1e6:.1f}M params")

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        features = self.backbone(x)
        fused = self.neck(features)
        return [self.head_p3(fused[0]), self.head_p4(fused[1]), self.head_p5(fused[2])]

    @classmethod
    def from_config(cls, config_path: str | Path) -> FDYOLO11:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
        channel_map = {"yolo11n": 32, "yolo11s": 48, "yolo11m": 64, "yolo11l": 80, "yolo11x": 96}
        base = model_cfg.get("base", "yolo11m")
        return cls(num_classes=model_cfg.get("num_classes", 6), base_channels=channel_map.get(base, 64))

    def export_onnx(self, save_path: str | Path, imgsz: int = 640, opset: int = 17) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.randn(1, 3, imgsz, imgsz, device=next(self.parameters()).device)
        self.eval()
        torch.onnx.export(self, dummy, str(save_path), opset_version=opset,
                          input_names=["images"], output_names=["output"])
        logger.info(f"ONNX exported: {save_path}")
        return save_path
