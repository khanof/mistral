FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install TTS dependencies
RUN pip install flask flask-cors edge-tts

# Set working directory
WORKDIR /app

# Copy API script
COPY tts_api.py /app/tts_api.py

# Expose API port
EXPOSE 5001

# Start API
CMD ["python3", "/app/tts_api.py"]
