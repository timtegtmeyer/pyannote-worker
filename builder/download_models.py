#!/usr/bin/env python3
"""
Pre-downloads pyannote models at image build time so cold starts are fast.
Requires HF_TOKEN env var with access to the gated pyannote models.
"""
import os
import sys
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not set — skipping model pre-download.")
    print("Models will be downloaded at runtime (slower cold start).")
    sys.exit(0)

models = [
    "pyannote/speaker-diarization-community-1",
]

for model_id in models:
    print(f"Downloading {model_id}...")
    snapshot_download(repo_id=model_id, token=hf_token)
    print(f"  Done: {model_id}")

print("All models downloaded successfully.")
