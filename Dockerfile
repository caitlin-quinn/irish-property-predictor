# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="L00196802@atu.ie"
LABEL description="Irish Property Price Predictor – Flask API"
LABEL version="1.0.0"

# Security: run as non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/   ./src/
COPY app/   ./app/
COPY models/ ./models/

# Train model at build time so it's baked into the image
RUN mkdir -p models data

RUN python src/train.py

RUN chown -R appuser:appuser /app

USER appuser

# Flask env
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONPATH=/app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app.app:app"]
