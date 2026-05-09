"""VLM Copilot API routes — Natural language visual queries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()

_copilot = None


def get_copilot():
    """Get or create the VLM copilot singleton."""
    global _copilot
    if _copilot is None:
        from src.vlm.copilot import VLMCopilot
        _copilot = VLMCopilot()
    return _copilot


class VLMQueryRequest(BaseModel):
    """Request body for VLM query."""
    query: str = Field(..., description="Natural language query about the image")
    max_tokens: int = Field(512, description="Maximum response tokens")


class VLMQueryResponse(BaseModel):
    """VLM query response."""
    query: str
    response: str
    latency_ms: float
    model: str


@router.post("/query", response_model=VLMQueryResponse)
async def query_vlm(
    image: UploadFile = File(..., description="Image to analyze"),
    query: str = Query(..., description="Natural language question"),
) -> VLMQueryResponse:
    """Ask a natural language question about an image.

    Examples:
    - "What defects are visible in this image?"
    - "Is this surface quality acceptable for shipping?"
    - "Describe the severity of the scratches"
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await image.read()
    try:
        import time
        copilot = get_copilot()
        start = time.perf_counter()
        response = copilot.query_with_image(contents, query)
        latency = (time.perf_counter() - start) * 1000
        model_name = copilot.model_name
    except ImportError:
        response = "This is a simulated response because the heavy ML libraries (transformers/torch) are not installed in this environment. I detected a significant scratch on the metal surface. The severity is high and requires immediate maintenance."
        latency = 120.5
        model_name = "qwen2.5-vl-mock"

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
    """Generate an automated defect inspection report from an image."""
    contents = await image.read()
    try:
        copilot = get_copilot()
        from src.vlm.report_gen import ReportGenerator
        generator = ReportGenerator(copilot)
        report = generator.generate_from_bytes(contents)
    except ImportError:
        import datetime
        report = {
            "report_id": f"RPT-MOCK-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "assessment": {
                "defect_found": True,
                "severity": "high",
                "raw_assessment": "Simulated Report: The surface shows a deep scratch measuring approximately 4cm in length. Rust formation is visible around the edges. This part fails quality assurance."
            },
            "pass_fail": "FAIL",
            "severity": "high"
        }

    return report
