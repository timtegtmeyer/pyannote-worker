# pyannote-worker

RunPod serverless worker for speaker diarization using
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
with a global ECAPA re-cluster pass on top.

## Input

```json
{
  "input": {
    "audio_url": "https://example.com/episode.mp3",
    "min_speakers": 2,
    "max_speakers": 4,
    "recluster_cosine_threshold": 0.25,
    "return_embeddings": true,
    "debug": false
  }
}
```

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `audio_url` | string | **required** | HTTPS URL to the episode audio (any format ffmpeg can read). |
| `min_speakers` | int | null | Hint to pyannote's clustering. Leave null to auto-detect. |
| `max_speakers` | int | null | Ditto. |
| `recluster_cosine_threshold` | float | 0.25 | Cosine *distance* below which two pyannote-produced speakers get merged in the global re-cluster pass. 0.25 = cosine similarity ≥ 0.75. |
| `return_embeddings` | bool | true | Whether to include per-speaker centroid vectors in the response. |
| `debug` | bool | false | Return diagnostics only, skip diarization. |

## Output

```json
{
  "output": {
    "segments": [
      { "speaker": "SPEAKER_00", "start": 5.2, "end": 12.8 },
      { "speaker": "SPEAKER_01", "start": 13.1, "end": 25.4 }
    ],
    "embeddings": {
      "SPEAKER_00": [0.123, -0.456, ...],
      "SPEAKER_01": [0.789, -0.012, ...]
    },
    "duration_sec": 3239.1,
    "overlap_sec": 42.5,
    "num_speakers_raw": 3,
    "num_speakers_merged": 2,
    "gpu_name": "NVIDIA RTX A5000",
    "diagnostics": {
      "decode_time_sec": 18.3,
      "diarization_time_sec": 221.4,
      "embedding_time_sec": 12.9,
      "num_segments": 512,
      "total_speech_sec": 2985.1,
      "recluster_threshold": 0.25,
      "recluster_remap": {"SPEAKER_00": "SPEAKER_00", "SPEAKER_02": "SPEAKER_00"}
    }
  }
}
```

`exclusive_speaker_diarization` is used for the segments, so every frame
is attributed to exactly one speaker — overlap regions are resolved by
pyannote internally.

## Pipeline

1. **Download** — fetch the audio over HTTPS.
2. **Decode + loudness-normalize** — ffmpeg converts to 16 kHz mono WAV
   with EBU R128 loudness normalization (I=-16 LUFS / TP=-1.5 dBTP / LRA=11).
   Matching the loudness target used when extracting speaker clips on the
   tenant-backend side keeps the ECAPA embedding space comparable across
   episodes.
3. **Diarize** — pyannote/speaker-diarization-community-1 runs VAD,
   WeSpeaker embedding, and VBx clustering in one pipeline.
4. **Per-speaker centroids** — pyannote/embedding extracts up to six
   segment embeddings per speaker (longest first, capped at 30 s each),
   averaged and L2-normalized.
5. **Global re-cluster** — agglomerative merge of speaker centroids
   whose cosine similarity ≥ `1 - recluster_cosine_threshold`. Folds the
   over-segmentation that pyannote occasionally produces on a single
   speaker whose level shifts mid-episode back into one label before the
   downstream ghost-speaker filter ever sees it.

## Setup

### 1. Accept model licenses on HuggingFace

- https://hf.co/pyannote/speaker-diarization-community-1
- https://hf.co/pyannote/embedding

### 2. Add repository secret

In GitHub → Settings → Secrets → Actions, add:

- `HF_TOKEN` — your HuggingFace access token (read permissions)

### 3. Build and push

```bash
make build
```

Increments the git tag, builds + pushes the image to GHCR with that
tag, and writes the digest to `.last-digest` for use in the RunPod
endpoint config.

### 4. Create RunPod endpoint

1. Go to https://www.runpod.io/console/serverless → New Endpoint
2. Choose **Custom** → enter the GHCR image URL (with the digest from
   `make digest`).
3. Set `HF_TOKEN` as an environment variable in the endpoint config.
4. Recommended GPU: A5000+ (the embedding pass needs ~3 GB VRAM on top
   of pyannote's ~4 GB).
5. Idle timeout: 5 s. Execution timeout: 600 s.

## Local testing

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=hf_... \
  -e RUNPOD_WEBHOOK_GET_JOB=... \
  ghcr.io/timtegtmeyer/pyannote-worker:latest
```
