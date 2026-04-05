"""
RunPod serverless handler for pyannote speaker diarization.

Expected input:
    { "audio_url": "https://example.com/episode.mp3" }

Output:
    {
        "segments": [
            { "speaker": "SPEAKER_00", "start": 5.2, "end": 12.8 },
            ...
        ]
    }
"""
import os
import tempfile
import logging

import requests
import runpod
import torch
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline — loaded once at container start
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None


def _load_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    hf_token = os.environ.get("HF_TOKEN")
    log.info("Loading pyannote/speaker-diarization-3.1 ...")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if torch.cuda.is_available():
        pipe = pipe.to(torch.device("cuda"))
        log.info("Pipeline moved to CUDA")
    else:
        log.warning("CUDA not available — running on CPU (will be slow)")

    _pipeline = pipe
    log.info("Pipeline ready")
    return _pipeline


# Pre-load at startup so the first job doesn't pay the load cost.
try:
    _load_pipeline()
except Exception as exc:
    log.error("Could not pre-load pipeline: %s", exc)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    audio_url = job_input.get("audio_url")

    if not audio_url:
        return {"error": "input.audio_url is required"}

    # Download audio to a temp file
    log.info("Downloading audio from %s", audio_url)
    try:
        response = requests.get(audio_url, stream=True, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"Failed to download audio: {exc}"}

    suffix = ".mp3"
    for ext in (".wav", ".flac", ".ogg", ".m4a", ".aac"):
        if audio_url.lower().split("?")[0].endswith(ext):
            suffix = ext
            break

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
            tmp_path = f.name

        log.info("Running diarization on %s ...", tmp_path)
        pipeline = _load_pipeline()
        diarization = pipeline(tmp_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

        log.info("Diarization complete: %d segments, %d speakers",
                 len(segments),
                 len({s["speaker"] for s in segments}))
        return {"segments": segments}

    except Exception as exc:
        log.exception("Diarization failed")
        return {"error": str(exc)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


runpod.serverless.start({"handler": handler})
