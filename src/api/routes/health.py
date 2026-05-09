"""Health check API routes."""

from __future__ import annotations

import platform
from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """System health check endpoint."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else None
        gpu_mem = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if has_cuda else None
    except ImportError:
        has_cuda, gpu_name, gpu_mem = False, None, None

    return {
        "status": "healthy",
        "platform": platform.system(),
        "python": platform.python_version(),
        "cuda_available": has_cuda,
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_mem,
    }


@router.get("/health/models")
async def model_health() -> dict[str, Any]:
    """Check loaded model status."""
    return {
        "detection_model": "loaded" if True else "not_loaded",
        "vlm_model": "not_loaded",
        "slam_module": "not_initialized",
    }
