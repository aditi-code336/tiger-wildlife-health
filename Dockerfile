# Production Dockerfile for Tiger Wildlife Health Diagnostic System
FROM python:3.11-slim

# Prevent python from writing pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install curl for health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application backend, frontend, models, and metadata
COPY app.py index.html tiger_diagnostics.py train_photo_nn.py preprocess.py model.joblib /app/
COPY models/ /app/models/
COPY data/prepared_data/ /app/data/prepared_data/
COPY data/predictions.db /app/data/predictions.db
COPY data/tiger_reserves_ntca.json /app/data/tiger_reserves_ntca.json

# Expose API and UI port
EXPOSE 8001

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Start Uvicorn server (respects $PORT environment variable on Render, Hugging Face, or local docker)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8001}"]
