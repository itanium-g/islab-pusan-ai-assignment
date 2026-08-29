# Multi-stage CUDA-Enabled PyTorch Container for Drone Detection
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual environment
RUN python3 -m venv /opt/venv
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . /workspace

# Set default command
CMD ["python", "train.py", "--config", "configs/model3_fpn_attn.yaml"]
