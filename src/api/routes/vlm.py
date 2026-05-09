"""VLM Copilot API routes — Natural language visual queries."""

from __future__ import annotations

import datetime
import io
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

router = APIRouter()


class VLMQueryResponse(BaseModel):
    """VLM query response."""
    query: str
    response: str
    latency_ms: float
    model: str


def _analyze_image(image_bytes: bytes) -> dict[str, Any]:
    """Analyze image using OpenCV — extract real visual features."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection for scratch/crack analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w)

    # Texture analysis via Laplacian variance (blur vs sharp)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Color distribution
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]

    # Dark region analysis (potential pitting/corrosion)
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = float(np.count_nonzero(dark_mask)) / (h * w)

    # Bright spot analysis (potential inclusion/reflection)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio = float(np.count_nonzero(bright_mask)) / (h * w)

    # Contour analysis for defect regions
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]

    return {
        "dimensions": f"{w}x{h}",
        "edge_density": round(edge_density, 4),
        "texture_sharpness": round(laplacian_var, 2),
        "mean_brightness": round(mean_v, 1),
        "mean_saturation": round(mean_s, 1),
        "dark_region_ratio": round(dark_ratio, 4),
        "bright_region_ratio": round(bright_ratio, 4),
        "num_edge_regions": len(significant_contours),
    }


def _generate_analysis_response(query: str, features: dict) -> str:
    """Generate a natural language response based on actual image features."""
    edge_density = features.get("edge_density", 0)
    sharpness = features.get("texture_sharpness", 0)
    dark_ratio = features.get("dark_region_ratio", 0)
    bright_ratio = features.get("bright_region_ratio", 0)
    num_regions = features.get("num_edge_regions", 0)
    brightness = features.get("mean_brightness", 128)

    # Determine defect indicators
    defects_found = []
    severity = "low"

    if edge_density > 0.15:
        defects_found.append("significant edge patterns suggesting scratches or cracks")
        severity = "high" if edge_density > 0.25 else "medium"

    if dark_ratio > 0.15:
        defects_found.append(f"dark regions covering {dark_ratio*100:.1f}% of the surface (potential pitting or corrosion)")
        severity = "high" if dark_ratio > 0.3 else max(severity, "medium")

    if bright_ratio > 0.1:
        defects_found.append(f"bright spots covering {bright_ratio*100:.1f}% (potential inclusion or surface anomalies)")

    if sharpness > 1000 and num_regions > 20:
        defects_found.append(f"highly textured surface with {num_regions} distinct edge regions (potential crazing or rolled-in scale)")

    if not defects_found:
        if brightness > 150 and edge_density < 0.05:
            return (
                f"Image Analysis ({features['dimensions']}): "
                f"The surface appears relatively clean with low edge density ({edge_density*100:.1f}%). "
                f"Mean brightness is {brightness:.0f}/255 with minimal dark regions ({dark_ratio*100:.1f}%). "
                f"No significant defect indicators detected. Surface quality appears acceptable for inspection."
            )
        defects_found.append(f"minor surface irregularities ({num_regions} edge regions detected)")

    defects_text = "; ".join(defects_found)
    return (
        f"Image Analysis ({features['dimensions']}): "
        f"Detected {defects_text}. "
        f"Edge density: {edge_density*100:.1f}%, texture sharpness index: {sharpness:.0f}, "
        f"dark coverage: {dark_ratio*100:.1f}%, bright coverage: {bright_ratio*100:.1f}%. "
        f"Overall severity assessment: {severity.upper()}. "
        f"Recommendation: {'Immediate maintenance required' if severity == 'high' else 'Schedule routine inspection' if severity == 'medium' else 'Continue monitoring'}."
    )


def _generate_report(features: dict) -> dict[str, Any]:
    """Generate a structured inspection report from image features."""
    edge_density = features.get("edge_density", 0)
    dark_ratio = features.get("dark_region_ratio", 0)

    # Determine severity
    if edge_density > 0.2 or dark_ratio > 0.25:
        severity = "high"
        pass_fail = "FAIL"
    elif edge_density > 0.1 or dark_ratio > 0.1:
        severity = "medium"
        pass_fail = "FAIL"
    elif edge_density > 0.05:
        severity = "low"
        pass_fail = "PASS"
    else:
        severity = "none"
        pass_fail = "PASS"

    analysis = _generate_analysis_response("report", features)

    return {
        "report_id": f"RPT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.datetime.now().isoformat(),
        "assessment": {
            "defect_found": severity not in ("none", "low"),
            "severity": severity,
            "raw_assessment": analysis,
        },
        "pass_fail": pass_fail,
        "severity": severity,
        "image_features": features,
    }


@router.post("/query", response_model=VLMQueryResponse)
async def query_vlm(
    image: UploadFile = File(..., description="Image to analyze"),
    query: str = Query(..., description="Natural language question"),
) -> VLMQueryResponse:
    """Ask a natural language question about an image.

    Uses computer vision analysis to provide data-driven responses
    based on actual image content (edge detection, texture analysis,
    color distribution, and contour analysis).
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await image.read()
    start = time.perf_counter()

    try:
        # Try full VLM model first
        from src.vlm.copilot import VLMCopilot
        copilot = VLMCopilot(device="cpu")
        response = copilot.query_with_image(contents, query)
        model_name = copilot.model_name
    except (ImportError, Exception):
        # Fallback: CV-based analysis with real image features
        features = _analyze_image(contents)
        response = _generate_analysis_response(query, features)
        model_name = "cv-analysis-engine"

    latency = (time.perf_counter() - start) * 1000
    logger.info(f"VLM query: '{query[:50]}...' latency={latency:.1f}ms model={model_name}")

    return VLMQueryResponse(
        query=query,
        response=response,
        latency_ms=round(latency, 2),
        model=model_name,
    )


@router.post("/report")
async def generate_report(
    image: UploadFile = File(..., description="Image for inspection report"),
) -> dict[str, Any]:
    """Generate an automated defect inspection report from an image.

    Analyzes the actual image content using edge detection, texture
    analysis, and color distribution to produce a data-driven report.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await image.read()

    try:
        from src.vlm.copilot import VLMCopilot
        copilot = VLMCopilot()
        from src.vlm.report_gen import ReportGenerator
        generator = ReportGenerator(copilot)
        report = generator.generate_from_bytes(contents)
    except (ImportError, Exception):
        features = _analyze_image(contents)
        report = _generate_report(features)

    return report
