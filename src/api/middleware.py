"""
API Middleware — Authentication, Logging, and CORS.
===================================================
Industrial-grade middleware for the FastAPI application.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import Request, Response
from loguru import logger
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "chigma_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "chigma_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and metrics collection."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency = time.perf_counter() - start_time

        # Record metrics
        endpoint = request.url.path
        method = request.method
        status = response.status_code

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

        # Log request
        logger.info(f"HTTP {method} {endpoint} -> Status: {status} | Latency: {latency * 1000:.2f}ms")

        return response

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Simple API Key Authentication Middleware for protected routes."""

    def __init__(self, app, api_keys: set[str], excluded_paths: list[str] = None):
        super().__init__(app)
        self.api_keys = api_keys
        self.excluded_paths = excluded_paths or ["/docs", "/openapi.json", "/metrics", "/api/v1/health"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.excluded_paths or any(request.url.path.startswith(p) for p in self.excluded_paths):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in self.api_keys:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API Key"})

        return await call_next(request)
