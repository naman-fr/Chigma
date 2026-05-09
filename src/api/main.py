"""
Chigma FastAPI Application — Production API Server
=====================================================
Main application with versioned routes, middleware, and
Prometheus metrics integration.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.routes.detection import router as detection_router
from src.api.routes.drone import router as drone_router
from src.api.routes.health import router as health_router
from src.api.routes.vlm import router as vlm_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    logger.info("Chigma API starting up...")
    # Load models, connect to services, etc.
    yield
    logger.info("Chigma API shutting down...")


app = FastAPI(
    title="Chigma — Industrial Vision & Drone Autonomy API",
    description=(
        "Production API for FD-YOLO11 defect detection, VLM copilot, "
        "and drone autonomy control. Part of the GenAI-Enhanced "
        "Industrial Vision platform."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

from src.api.middleware import RequestLoggingMiddleware

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Logging & Metrics Middleware ──
app.add_middleware(RequestLoggingMiddleware)



# ── Routes ──
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(detection_router, prefix="/api/v1/detection", tags=["Detection"])
app.include_router(vlm_router, prefix="/api/v1/vlm", tags=["Vision-Language"])
app.include_router(drone_router, prefix="/api/v1/drone", tags=["Drone Autonomy"])


# ── Prometheus Metrics Endpoint ──
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


import os

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# ── Root ──
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Chigma Frontend Dashboard."""
    template_path = os.path.join("src", "api", "templates", "index.html")
    with open(template_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── CLI Entry Point ──
def cli():
    """Run the Chigma API server."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
