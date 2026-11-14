# ==============================================================================
# Dockerfile — Indian Sign Language Recognition & Translation Backend
# Production Python 3.10 Container with OpenCV and MediaPipe system libraries
# ==============================================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies required for OpenCV, MediaPipe, and Audio TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and public assets
COPY models/ models/
COPY src/ src/
COPY public/ public/
COPY run_app.sh .

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/api/status || exit 1

CMD ["python", "src/app.py"]
