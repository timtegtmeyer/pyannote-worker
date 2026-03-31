FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: Python 3.10, ffmpeg (required by pyannote/torchaudio), wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg wget git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY builder/requirements.txt /builder/requirements.txt
RUN pip3 install --no-cache-dir -r /builder/requirements.txt

# Pre-download pyannote models at build time using the HF_TOKEN secret.
# The models land in /root/.cache/huggingface/ and are baked into the image
# so cold starts are fast (no download at runtime).
COPY builder/download_models.py /builder/download_models.py
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python3 /builder/download_models.py

# Copy handler
COPY src/ /app/

CMD ["python3", "-u", "handler.py"]
