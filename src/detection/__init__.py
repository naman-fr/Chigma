"""
Chigma Detection Module — FD-YOLO11 Industrial Defect Detection
================================================================
Implements Feature-Enhanced YOLO11 with:
- SC-C3k2: Self-Calibrated Convolution blocks
- FSPPF: Feature-fusion Spatial Pyramid Pooling Fast
- DySample: Dynamic upsampling mechanism

Reference: "FD-YOLO11: Feature-Enhanced Deep Learning for Steel Surface
Defect Detection" (IEEE Access, 2025)
"""

from src.detection.dysample import DySample
from src.detection.fd_yolo11 import FDYOLO11
from src.detection.fsppf import FSPPF
from src.detection.sc_c3k2 import SCC3k2

__all__ = ["FDYOLO11", "SCC3k2", "FSPPF", "DySample"]
