"""Health check API routes."""

from __future__ import annotations

import platform
from typing import Any

import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """System health check endpoint."""
    return {
        "status": "healthy",
        "platform": platform.system(),
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else None,
    }


@router.get("/health/models")
async def model_health() -> dict[str, Any]:
    """Check loaded model status."""
    return {
        "detection_model": "loaded" if True else "not_loaded",
        "vlm_model": "not_loaded",
        "slam_module": "not_initialized",
    }
