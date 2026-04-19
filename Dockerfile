FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app.py .
COPY audio.py .
COPY audio_narration.py .
COPY frame_model.keras .
COPY alphabet_model.keras .
COPY model/ model/
COPY templates/ templates/
COPY static_frontend/ static_frontend/

# Expose port
EXPOSE 10000

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=10000

# Start command with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
