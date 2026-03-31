# pyannote-worker

RunPod serverless worker for speaker diarization using [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Input

```json
{ "input": { "audio_url": "https://example.com/episode.mp3" } }
```

## Output

```json
{
  "output": {
    "segments": [
      { "speaker": "SPEAKER_00", "start": 5.2, "end": 12.8 },
      { "speaker": "SPEAKER_01", "start": 13.1, "end": 25.4 }
    ]
  }
}
```

## Setup

### 1. Accept model licenses on HuggingFace

- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0

### 2. Add repository secret

In GitHub → Settings → Secrets → Actions, add:

- `HF_TOKEN` — your HuggingFace access token (read permissions)

### 3. Push to main to trigger the build

The GitHub Actions workflow builds the image and pushes it to:

```
ghcr.io/timtegtmeyer/pyannote-worker:latest
```

### 4. Create RunPod endpoint

1. Go to https://www.runpod.io/console/serverless → New Endpoint
2. Choose **Custom** → enter the GHCR image URL
3. Set `HF_TOKEN` as an environment variable in the endpoint config
4. Recommended GPU: Tesla T4 | Idle timeout: 5s | Max workers: 1

## Local testing

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=hf_... \
  -e RUNPOD_WEBHOOK_GET_JOB=... \
  ghcr.io/timtegtmeyer/pyannote-worker:latest
```
