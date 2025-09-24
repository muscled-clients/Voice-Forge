# Multi-stage build for production
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash voiceforge

WORKDIR /app

# Development stage
FROM base AS development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

# Copy application code
COPY --chown=voiceforge:voiceforge . .

USER voiceforge

# Development command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production build stage
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --user -r requirements.txt

# Production stage
FROM base AS production

# Copy Python packages from builder
COPY --from=builder /root/.local /home/voiceforge/.local

# Copy application code
COPY --chown=voiceforge:voiceforge ./src /app/src
COPY --chown=voiceforge:voiceforge ./models /app/models

# Set PATH for user-installed packages
ENV PATH=/home/voiceforge/.local/bin:$PATH

USER voiceforge

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]