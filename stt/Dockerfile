FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install STT dependencies
RUN pip install flask flask-cors openai-whisper torch

# Set working directory
WORKDIR /app

# Copy API script
COPY stt_api.py /app/stt_api.py
RUN pip install scipy>=1.8.0
# Expose API port
EXPOSE 5002
RUN pip install flask-socketio numpy
# Start API
CMD ["python3", "/app/stt_api.py"]
