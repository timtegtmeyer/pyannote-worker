"""
RunPod serverless handler for pyannote speaker diarization.

Expected input:
    { "audio_url": "https://example.com/episode.mp3" }

    Optional debug mode (returns diagnostics without running diarization):
    { "debug": true }

Output:
    {
        "segments": [
            { "speaker": "SPEAKER_00", "start": 5.2, "end": 12.8 },
            ...
        ]
    }
"""
import os
import sys
import time
import tempfile
import logging

import requests
import runpod
import torch
from pyannote.audio import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("pyannote-worker")

# ---------------------------------------------------------------------------
# Pipeline — loaded once at container start
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None
_pipeline_load_error: str | None = None
_pipeline_load_time: float | None = None


def _load_pipeline() -> Pipeline:
    global _pipeline, _pipeline_load_error, _pipeline_load_time
    if _pipeline is not None:
        return _pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        _pipeline_load_error = "HF_TOKEN environment variable is not set"
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it in RunPod template settings -> Environment Variables."
        )

    try:
        from pyannote.audio import __version__ as pa_version
    except ImportError:
        pa_version = "unknown"
    log.info("Loading pyannote/speaker-diarization-3.1 (pyannote.audio=%s, HF_TOKEN: %s...%s)",
             pa_version, hf_token[:5], hf_token[-4:])

    start = time.time()
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as exc:
        _pipeline_load_error = f"Pipeline.from_pretrained failed: {exc}"
        log.exception("Failed to load pipeline")
        raise

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipe = pipe.to(device)
        log.info("Pipeline moved to CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        log.warning("CUDA not available — running on CPU (will be slow)")

    elapsed = time.time() - start
    _pipeline_load_time = elapsed
    _pipeline = pipe
    log.info("Pipeline ready (loaded in %.1fs)", elapsed)
    return _pipeline


# Pre-load at startup so the first job doesn't pay the load cost.
try:
    _load_pipeline()
except Exception as exc:
    log.error("Could not pre-load pipeline: %s", exc)


# ---------------------------------------------------------------------------
# Debug / health-check handler
# ---------------------------------------------------------------------------

def _debug_info() -> dict:
    """Return diagnostic info without running diarization."""
    hf_token = os.environ.get("HF_TOKEN")
    cuda_available = torch.cuda.is_available()

    info = {
        "status": "ok" if _pipeline is not None else "error",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_memory_gb": round(getattr(torch.cuda.get_device_properties(0), 'total_memory', getattr(torch.cuda.get_device_properties(0), 'total_mem', 0)) / 1e9, 1) if cuda_available else None,
        "hf_token_set": bool(hf_token),
        "hf_token_preview": f"{hf_token[:5]}...{hf_token[-4:]}" if hf_token else None,
        "pipeline_loaded": _pipeline is not None,
        "pipeline_load_error": _pipeline_load_error,
        "pipeline_load_time_sec": round(_pipeline_load_time, 1) if _pipeline_load_time else None,
    }

    # Try a quick import check for pyannote models
    try:
        from pyannote.audio import __version__ as pyannote_version
        info["pyannote_version"] = pyannote_version
    except Exception:
        info["pyannote_version"] = "unknown"

    return info


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    log.info("[job:%s] Received job with input keys: %s", job_id, list(job_input.keys()))

    # Debug mode: return diagnostics
    if job_input.get("debug"):
        log.info("[job:%s] Debug mode requested", job_id)
        return _debug_info()

    audio_url = job_input.get("audio_url")

    if not audio_url:
        log.error("[job:%s] Missing audio_url in input", job_id)
        return {"error": "input.audio_url is required"}

    # Step 1: Download audio
    log.info("[job:%s] Downloading audio from %s", job_id, audio_url)
    download_start = time.time()
    try:
        response = requests.get(audio_url, stream=True, timeout=300)
        response.raise_for_status()
        content_length = response.headers.get("Content-Length", "unknown")
        content_type = response.headers.get("Content-Type", "unknown")
        log.info("[job:%s] Download response: status=%d, content-length=%s, content-type=%s",
                 job_id, response.status_code, content_length, content_type)
    except requests.RequestException as exc:
        log.error("[job:%s] Failed to download audio: %s", job_id, exc)
        return {"error": f"Failed to download audio: {exc}"}

    suffix = ".mp3"
    for ext in (".wav", ".flac", ".ogg", ".m4a", ".aac"):
        if audio_url.lower().split("?")[0].endswith(ext):
            suffix = ext
            break

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            bytes_written = 0
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                bytes_written += len(chunk)
            tmp_path = f.name

        download_elapsed = time.time() - download_start
        log.info("[job:%s] Audio saved to %s (%.1f MB in %.1fs)",
                 job_id, tmp_path, bytes_written / 1e6, download_elapsed)

        # Step 2: Run diarization
        log.info("[job:%s] Loading pipeline...", job_id)
        pipeline = _load_pipeline()

        log.info("[job:%s] Running diarization on %s ...", job_id, tmp_path)
        diarize_start = time.time()
        diarization = pipeline(tmp_path)
        diarize_elapsed = time.time() - diarize_start

        # Step 3: Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

        unique_speakers = {s["speaker"] for s in segments}
        total_speech_sec = sum(s["end"] - s["start"] for s in segments)

        log.info("[job:%s] Diarization complete in %.1fs: %d segments, %d speakers, %.1fs total speech",
                 job_id, diarize_elapsed, len(segments), len(unique_speakers), total_speech_sec)

        return {
            "segments": segments,
            "diagnostics": {
                "download_time_sec": round(download_elapsed, 1),
                "diarization_time_sec": round(diarize_elapsed, 1),
                "audio_size_mb": round(bytes_written / 1e6, 1),
                "num_segments": len(segments),
                "num_speakers": len(unique_speakers),
                "total_speech_sec": round(total_speech_sec, 1),
            },
        }

    except Exception as exc:
        log.exception("[job:%s] Diarization failed", job_id)
        return {"error": str(exc)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


runpod.serverless.start({"handler": handler})
