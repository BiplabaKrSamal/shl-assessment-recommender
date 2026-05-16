# ─────────────────────────────────────────────────────────────────────────────
# SHL Assessment Recommender — Dockerfile
#
# Build strategy:
#   Stage 1 (builder): Install Python deps into /install (avoids re-downloading
#                      on each build by exploiting Docker layer cache).
#   Stage 2 (runtime): Slim image with only what's needed to run.
#
# Cold-start optimization:
#   - sentence-transformers model (all-MiniLM-L6-v2, ~22MB) is downloaded at
#     BUILD TIME into /app/model_cache. This means the free-tier Render dyno
#     wakes up cold but doesn't need to download the model — just loads from disk.
#   - FAISS index is built in-memory at startup (~2-3s for 60 items). Too small
#     to pre-build; simpler to rebuild on each cold start.
#
# Image size target: ~900MB (sentence-transformers + torch CPU dominate).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /install

# System deps needed for lxml, faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/packages --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Runtime stage
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Pre-download sentence-transformer model at build time
# This avoids downloading on cold start (critical for 30s timeout compliance)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/model_cache'); \
print('Model cached at /app/model_cache')"

# Set model cache env so retriever finds it
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV HF_HOME=/app/model_cache

# Application config
ENV CATALOG_PATH=/app/data/catalog.json
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check — allows 2 minutes for cold start per spec
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); exit(0 if r.status_code==200 else 1)"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
