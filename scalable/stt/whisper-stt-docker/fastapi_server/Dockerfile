FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including libsndfile and ffmpeg
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Show pip installation progress
ENV PIP_PROGRESS_BAR=on
ENV PIP_NO_CACHE_DIR=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install numpy first
RUN pip install numpy==1.23.5

# Then install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create static directory
RUN mkdir -p static

# Make sure the script is executable
RUN chmod +x /app/main.py

# Explicitly set host and port with log-level for debugging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"] 