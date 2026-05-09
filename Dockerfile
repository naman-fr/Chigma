# ============================================================================
# Chigma — Production Dockerfile
# Multi-stage build for optimized image size
# ============================================================================

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    pip install --no-cache-dir ".[dev]" --target /install

# ── Stage 2: Runtime ──
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/dist-packages

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY mlops/ /app/mlops/

# Create directories
RUN mkdir -p /app/models/checkpoints /app/models/exports /app/data /app/logs /app/metrics

# Non-root user for security
RUN groupadd -r chigma && useradd -r -g chigma chigma && \
    chown -R chigma:chigma /app
USER chigma

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
