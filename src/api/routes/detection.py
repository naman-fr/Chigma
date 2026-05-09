"""Detection API routes — FD-YOLO11 inference endpoints."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

# Lazy-loaded inference engine
_engine = None


def get_engine():
    """Get or create the inference engine singleton."""
    global _engine
    if _engine is None:
        from src.detection.inference import InferenceEngine
        model_path = "models/checkpoints/fd-yolo11/weights/best.pt"
        if not Path(model_path).exists():
            model_path = "yolo11n.pt"  # Fallback to pretrained
        _engine = InferenceEngine(model_path)
    return _engine


class DetectionResult(BaseModel):
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")


class DetectionResponse(BaseModel):
    """Full detection response."""
    detections: list[DetectionResult]
    num_detections: int
    latency_ms: float
    image_shape: list[int]


@router.post("/predict", response_model=DetectionResponse)
async def predict(
    image: UploadFile = File(..., description="Image to analyze for defects"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS"),
) -> DetectionResponse:
    """Run FD-YOLO11 defect detection on an uploaded image.

    Returns bounding boxes, class labels, and confidence scores
    for all detected surface defects.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)

    import cv2
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    engine = get_engine()
    engine.conf_threshold = conf
    engine.iou_threshold = iou

    result = engine.predict(img)
    return DetectionResponse(**result)


@router.post("/predict/batch")
async def predict_batch(
    images: list[UploadFile] = File(..., description="Batch of images"),
) -> list[DetectionResponse]:
    """Batch inference on multiple images."""
    import cv2

    results = []
    engine = get_engine()

    for img_file in images:
        contents = await img_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            result = engine.predict(img)
            results.append(DetectionResponse(**result))

    return results


@router.get("/model/info")
async def model_info() -> dict[str, Any]:
    """Get current model information."""
    engine = get_engine()
    return {
        "model_path": str(engine.model_path),
        "device": engine.device,
        "conf_threshold": engine.conf_threshold,
        "iou_threshold": engine.iou_threshold,
        "classes": engine.DEFECT_CLASSES,
    }
