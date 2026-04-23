# pyannote.audio 4.x pulls in torch==2.8.0 via its dependency chain (torchcodec,
# torchaudio) and replaces the torch in the base image. torchvision 0.20 that
# shipped with pytorch 2.5.1 is then incompatible with the new torch, causing a
# circular-import crash at handler startup. Match the base image to torch 2.8
# so torch + torchvision + torchaudio stay on the same major+minor.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: ffmpeg (required by pyannote/torchaudio), wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wget git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY builder/requirements.txt /builder/requirements.txt
RUN pip install --no-cache-dir -r /builder/requirements.txt

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
