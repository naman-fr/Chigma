# ============================================================================
# Chigma — Production Dockerfile
# Multi-stage build for optimized image size
# ============================================================================

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only the runtime API dependencies (not the full ML stack)
# ML/DL libraries (torch, ultralytics, transformers) are installed
# separately in GPU-enabled deployment environments.
COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi "uvicorn[standard]" pydantic python-multipart aiofiles \
        prometheus-client pyyaml rich loguru click tqdm jinja2 httpx \
        numpy opencv-python-headless pillow scipy \
        --target /install

# ── Stage 2: Runtime ──
FROM python:3.11-slim

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Create directories
RUN mkdir -p /app/models/checkpoints /app/models/exports /app/data /app/logs /app/metrics

# Non-root user for security
RUN groupadd -r chigma && useradd -r -g chigma chigma && \
    chown -R chigma:chigma /app
USER chigma

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
