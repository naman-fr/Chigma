"""VLM Copilot API routes — Natural language visual queries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
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
    copilot = get_copilot()

    import time
    start = time.perf_counter()
    response = copilot.query_with_image(contents, query)
    latency = (time.perf_counter() - start) * 1000

    return VLMQueryResponse(
        query=query,
        response=response,
        latency_ms=round(latency, 2),
        model=copilot.model_name,
    )


@router.post("/report")
async def generate_report(
    image: UploadFile = File(..., description="Image for inspection report"),
) -> dict[str, Any]:
    """Generate an automated defect inspection report from an image."""
    contents = await image.read()
    copilot = get_copilot()

    from src.vlm.report_gen import ReportGenerator
    generator = ReportGenerator(copilot)
    report = generator.generate_from_bytes(contents)

    return report
