# Use NVIDIA CUDA image for GPU acceleration
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set non-interactive mode for package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0  

# Install necessary system dependencies (including Rust for tokenizers)
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Rust (required for tokenizers)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies (pinned for stability)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    vllm==0.2.7 \
    numpy

# Install autoawq separately to avoid missing dependencies
RUN pip3 install --no-cache-dir autoawq

RUN pip install --upgrade transformers

# Prevent tokenizers from running in parallel (fixes hanging issues)
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Define a model mount point (this ensures the folder is not erased on rebuild)
RUN mkdir -p /app/models/mistralai/Mistral-7B-Instruct-v0.1-AWQ

# Copy API script into the container
COPY api.py /app/api.py

# Expose API ports
EXPOSE 5000 8001

# **CMD now downloads the model at runtime if missing**
CMD ["/bin/bash", "-c", " \
    if [ ! -f /app/models/mistralai/Mistral-7B-Instruct-v0.1-AWQ/config.json ]; then \
        echo 'Downloading Mistral-7B AWQ model...'; \
        python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"TheBloke/Mistral-7B-Instruct-v0.1-AWQ\", local_dir=\"/app/models/mistralai/Mistral-7B-Instruct-v0.1-AWQ\", token=None)'; \
    else \
        echo 'Model already exists. Skipping download.'; \
    fi && \
    python3 -m vllm.entrypoints.api_server \
    --model /app/models/mistralai/Mistral-7B-Instruct-v0.1-AWQ \
    --quantization awq \
     --dtype float16 \
    --gpu-memory-utilization 1.0 \
    --max-model-len 4096 \
    --port 8001"]
