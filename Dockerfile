# ==============================================================================
# ShieldML — Production Container Image
# ==============================================================================
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and package metadata
COPY pyproject.toml setup.py README.md .
COPY shieldml/ shieldml/
COPY scripts/ scripts/
COPY frontend/ frontend/
COPY data/ data/
COPY artifacts/ artifacts/

# Install shieldml package in editable mode
RUN pip install --no-cache-dir -e .

# Expose FastAPI HTTP serving port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch high-performance Uvicorn server
CMD ["uvicorn", "shieldml.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
