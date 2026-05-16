# ─────────────────────────────────────────────────────────────────────────────
# SHL Assessment Recommender — Dockerfile
#
# Designed for Render.com free tier (512MB RAM).
# Uses scikit-learn TF-IDF instead of sentence-transformers+torch.
# Total image size: ~250MB (vs ~900MB with torch).
# Cold start: ~2s (TF-IDF matrix fit on 57 items).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY data/ ./data/

# Config
ENV CATALOG_PATH=/app/data/catalog.json
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8000/health',timeout=8); exit(0 if r.status==200 else 1)"

# Shell form so $PORT env var is expanded (required for Render.com)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30 --log-level info
