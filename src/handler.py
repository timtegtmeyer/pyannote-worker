"""
RunPod serverless handler for pyannote speaker diarization.

Pipeline:
  1. Download audio
  2. Decode to 16 kHz mono WAV + EBU R128 loudness normalization (via ffmpeg)
  3. Run pyannote/speaker-diarization-community-1 (VAD + VBx clustering +
     WeSpeaker embeddings, all bundled) and take the exclusive track so the
     downstream word→speaker lookup never sees overlap frames.
  4. Extract one pyannote/embedding vector per speaker by averaging the
     embeddings of that speaker's longest N segments — this is what the
     consumer re-clusters against and what tenant-backend compares with
     stored speaker profiles.
  5. Global re-cluster: agglomerative merge of speaker centroids whose
     cosine distance falls below recluster_cosine_threshold (default 0.25,
     i.e. similarity ≥ 0.75). Pyannote tends to over-segment a single
     speaker across long-form audio when levels shift; this folds those
     duplicates back together before the ghost-speaker filter in
     tenant-backend ever sees them.

Expected input:
    {
        "audio_url": "https://example.com/episode.mp3",
        "min_speakers": int | None,
        "max_speakers": int | None,
        "recluster_cosine_threshold": float (default 0.25),
        "return_embeddings": bool (default true),
        "debug": bool (default false)
    }

Output:
    {
        "segments": [{"speaker": "SPEAKER_00", "start": 5.2, "end": 12.8}, ...],
        "embeddings": {"SPEAKER_00": [512 floats], ...} (if return_embeddings),
        "duration_sec": 3239.1,
        "overlap_sec": 42.5,
        "num_speakers_raw": 4,
        "num_speakers_merged": 2,
        "gpu_name": "NVIDIA RTX A5000",
        "diagnostics": {...}
    }
"""
import os
import sys
import time
import tempfile
import logging
import subprocess
from collections import defaultdict

import numpy as np
import requests
import runpod
import torch


def _gpu_name() -> str | None:
    """Return the GPU's display name (e.g. 'NVIDIA GeForce RTX 4090') or None.
    Saas-side cost calculation looks this up in the RunPod USD/hr map so
    it charges the exact rate of the GPU the job actually ran on, instead
    of guessing from the endpoint's gpuIds ladder."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


# PyTorch 2.8+ defaults torch.load to weights_only=True, which breaks
# pyannote's model loading. Allow the TorchVersion global used in checkpoints.
try:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except (AttributeError, TypeError):
    pass

from pyannote.audio import Pipeline, Inference, Model
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.core import Segment

# PyTorch 2.6+ defaults torch.load to weights_only=True, which rejects
# pyannote checkpoint globals. Allowlist the classes stored in checkpoints.
torch.serialization.add_safe_globals([Specifications, Problem, Resolution])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("pyannote-worker")

# ---------------------------------------------------------------------------
# Models — loaded once at container start
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None
_embedder: Inference | None = None
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
    log.info(
        "Loading pyannote/speaker-diarization-community-1 (pyannote.audio=%s, HF_TOKEN: %s...%s)",
        pa_version, hf_token[:5], hf_token[-4:],
    )

    start = time.time()
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
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


def _load_embedder() -> Inference:
    """Standalone ECAPA-style embedding model used for the global re-cluster
    pass. Community-1 embeds internally but doesn't expose per-segment
    vectors, so we run pyannote/embedding once more per cluster — cheap
    compared to the diarization pipeline itself."""
    global _embedder
    if _embedder is not None:
        return _embedder

    hf_token = os.environ.get("HF_TOKEN")
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # window="whole" takes one vector over the supplied Segment instead of a
    # sliding window — matches what we want (one embedding per turn).
    _embedder = Inference(model, window="whole", device=device)
    log.info("Embedding model ready on %s", device)
    return _embedder


# Pre-load at startup so the first job doesn't pay the load cost.
try:
    _load_pipeline()
    _load_embedder()
except Exception as exc:
    log.error("Could not pre-load models: %s", exc)


# ---------------------------------------------------------------------------
# Debug / health-check handler
# ---------------------------------------------------------------------------

def _debug_info() -> dict:
    hf_token = os.environ.get("HF_TOKEN")
    cuda_available = torch.cuda.is_available()

    info = {
        "status": "ok" if _pipeline is not None else "error",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "hf_token_set": bool(hf_token),
        "hf_token_preview": f"{hf_token[:5]}...{hf_token[-4:]}" if hf_token else None,
        "pipeline_loaded": _pipeline is not None,
        "embedder_loaded": _embedder is not None,
        "pipeline_load_error": _pipeline_load_error,
        "pipeline_load_time_sec": round(_pipeline_load_time, 1) if _pipeline_load_time else None,
    }

    try:
        from pyannote.audio import __version__ as pyannote_version
        info["pyannote_version"] = pyannote_version
    except Exception:
        info["pyannote_version"] = "unknown"

    return info


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ffmpeg_decode(src_path: str, dst_path: str) -> None:
    """Decode any input → 16 kHz mono WAV with EBU R128 loudness
    normalization. Running loudnorm once here keeps the VAD decision
    boundary, cluster assignment, and embedding space on a consistent
    loudness target — essential for the re-cluster step and for matching
    stored speaker profiles (which are clipped from audio that went
    through the same normalization on the tenant-backend side)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "16000", "-ac", "1", "-f", "wav", dst_path,
        ],
        check=True, capture_output=True,
    )


def _audio_duration_sec(wav_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def _compute_embedding(wav_path: str, seg: Segment) -> np.ndarray | None:
    """Extract one embedding over the given segment. Returns an L2-
    normalized numpy vector, or None on failure (segment too short for
    the embedder's minimum window, decode error, etc.)."""
    if seg.duration < 0.3:
        # pyannote/embedding's sliding minimum is ~0.5 s but "whole"
        # window accepts short inputs and zero-pads; 0.3 s floor keeps
        # the noise of micro-segments out of the centroid.
        return None
    try:
        embedder = _load_embedder()
        vec = embedder.crop(wav_path, seg)
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception:
        return None


def _speaker_centroids(
    wav_path: str,
    annotation,
    max_segments_per_speaker: int = 6,
) -> dict[str, np.ndarray]:
    """Average the top-N longest segment embeddings per speaker into a
    centroid. Six samples × up-to-30 s each is enough for a stable ECAPA
    mean without paying for the full cluster, which on a 1 h episode with
    ~800 diarized segments would otherwise add 20-40 s to the job."""
    by_speaker: dict[str, list[Segment]] = defaultdict(list)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        by_speaker[speaker].append(turn)

    centroids: dict[str, np.ndarray] = {}
    for speaker, segs in by_speaker.items():
        segs.sort(key=lambda s: s.duration, reverse=True)
        vecs = []
        for seg in segs[:max_segments_per_speaker]:
            clipped = Segment(seg.start, min(seg.end, seg.start + 30.0))
            v = _compute_embedding(wav_path, clipped)
            if v is not None:
                vecs.append(v)
        if vecs:
            mean = np.mean(vecs, axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                mean = mean / norm
            centroids[speaker] = mean

    return centroids


def _recluster(
    centroids: dict[str, np.ndarray],
    cosine_threshold: float,
) -> dict[str, str]:
    """Greedy agglomerative merge: walk pairs of centroids from highest
    similarity down, merging whenever 1 - cosine_distance ≥ threshold.
    Returns a label → canonical-label map. Pure Python because the
    cluster count is small (typically 2-6)."""
    if len(centroids) <= 1 or cosine_threshold >= 1.0:
        return {label: label for label in centroids}

    labels = list(centroids.keys())
    # Union-find over speaker labels
    parent = {lab: lab for lab in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the lexicographically smaller label as root (stable output)
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    pairs = []
    for i, a in enumerate(labels):
        va = centroids[a]
        for b in labels[i + 1:]:
            vb = centroids[b]
            sim = float(np.dot(va, vb))
            pairs.append((sim, a, b))

    pairs.sort(reverse=True)
    merged = []
    for sim, a, b in pairs:
        if sim < (1.0 - cosine_threshold):
            break
        if find(a) != find(b):
            union(a, b)
            merged.append((a, b, sim))

    if merged:
        log.info("Recluster merges: %s", merged)

    return {lab: find(lab) for lab in labels}


def _apply_remap(segments: list[dict], remap: dict[str, str]) -> list[dict]:
    if not remap:
        return segments
    for seg in segments:
        seg["speaker"] = remap.get(seg["speaker"], seg["speaker"])
    return segments


def _measure_overlap_sec(diarization) -> float:
    """Total duration covered by >1 speaker in the raw (non-exclusive)
    annotation. Useful diagnostic — a podcast with heavy crosstalk may
    warrant a lower recluster threshold or human review."""
    try:
        overlap = diarization.get_overlap()
        return float(sum(s.duration for s in overlap))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    log.info("[job:%s] Received job with input keys: %s", job_id, list(job_input.keys()))

    if job_input.get("debug"):
        log.info("[job:%s] Debug mode requested", job_id)
        return _debug_info()

    audio_url = job_input.get("audio_url")
    if not audio_url:
        log.error("[job:%s] Missing audio_url in input", job_id)
        return {"error": "input.audio_url is required"}

    min_speakers = job_input.get("min_speakers")
    max_speakers = job_input.get("max_speakers")
    recluster_threshold = float(job_input.get("recluster_cosine_threshold", 0.25))
    return_embeddings = bool(job_input.get("return_embeddings", True))

    # Step 1: Download
    log.info("[job:%s] Downloading audio from %s", job_id, audio_url)
    download_start = time.time()
    try:
        response = requests.get(audio_url, stream=True, timeout=300)
        response.raise_for_status()
        content_length = response.headers.get("Content-Length", "unknown")
        content_type = response.headers.get("Content-Type", "unknown")
        log.info(
            "[job:%s] Download response: status=%d, content-length=%s, content-type=%s",
            job_id, response.status_code, content_length, content_type,
        )
    except requests.RequestException as exc:
        log.error("[job:%s] Failed to download audio: %s", job_id, exc)
        return {"error": f"Failed to download audio: {exc}"}

    suffix = ".mp3"
    for ext in (".wav", ".flac", ".ogg", ".m4a", ".aac"):
        if audio_url.lower().split("?")[0].endswith(ext):
            suffix = ext
            break

    tmp_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            bytes_written = 0
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                bytes_written += len(chunk)
            tmp_path = f.name

        download_elapsed = time.time() - download_start
        log.info(
            "[job:%s] Audio saved to %s (%.1f MB in %.1fs)",
            job_id, tmp_path, bytes_written / 1e6, download_elapsed,
        )

        # Step 2: Decode + loudnorm
        wav_path = tmp_path.rsplit(".", 1)[0] + ".norm.wav"
        log.info("[job:%s] Decoding + loudness-normalizing to WAV...", job_id)
        convert_start = time.time()
        _ffmpeg_decode(tmp_path, wav_path)
        convert_elapsed = time.time() - convert_start
        log.info("[job:%s] Decoded in %.1fs", job_id, convert_elapsed)

        duration_sec = _audio_duration_sec(wav_path)

        # Step 3: Diarize
        pipeline = _load_pipeline()
        log.info(
            "[job:%s] Running diarization (min_speakers=%s max_speakers=%s) ...",
            job_id, min_speakers, max_speakers,
        )
        diarize_start = time.time()
        diar_kwargs = {}
        if isinstance(min_speakers, int) and min_speakers > 0:
            diar_kwargs["min_speakers"] = min_speakers
        if isinstance(max_speakers, int) and max_speakers > 0:
            diar_kwargs["max_speakers"] = max_speakers
        diarization = pipeline(wav_path, **diar_kwargs)
        diarize_elapsed = time.time() - diarize_start

        overlap_sec = _measure_overlap_sec(diarization)

        # Exclusive variant: one speaker per frame. Downstream word→speaker
        # attribution is a single-speaker lookup and mixed-overlap regions
        # would otherwise get an arbitrary label.
        annotation = getattr(diarization, "exclusive_speaker_diarization", diarization)

        segments_raw = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments_raw.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

        raw_speakers = {s["speaker"] for s in segments_raw}
        log.info(
            "[job:%s] Diarization complete in %.1fs: %d segments, %d raw speakers, %.1fs overlap",
            job_id, diarize_elapsed, len(segments_raw), len(raw_speakers), overlap_sec,
        )

        # Step 4: per-speaker centroids + global re-cluster
        embed_start = time.time()
        centroids = _speaker_centroids(wav_path, annotation)
        embed_elapsed = time.time() - embed_start
        log.info(
            "[job:%s] Extracted %d speaker centroids in %.1fs",
            job_id, len(centroids), embed_elapsed,
        )

        remap = _recluster(centroids, recluster_threshold)
        segments = _apply_remap(segments_raw, remap)

        # Merge adjacent same-speaker segments created by the remap. Anti-
        # overlap property is preserved because the inputs were already
        # non-overlapping (exclusive_speaker_diarization) and we only ever
        # rename, never move, boundaries.
        merged_segments: list[dict] = []
        for seg in segments:
            if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"] \
                    and seg["start"] - merged_segments[-1]["end"] < 0.5:
                merged_segments[-1]["end"] = seg["end"]
            else:
                merged_segments.append(dict(seg))
        segments = merged_segments

        # Fold centroids into canonical labels for the output payload
        canonical_centroids: dict[str, list[float]] = {}
        if return_embeddings and centroids:
            grouped: dict[str, list[np.ndarray]] = defaultdict(list)
            for label, vec in centroids.items():
                grouped[remap.get(label, label)].append(vec)
            for canonical, vecs in grouped.items():
                mean = np.mean(np.stack(vecs), axis=0)
                norm = float(np.linalg.norm(mean))
                if norm > 0:
                    mean = mean / norm
                canonical_centroids[canonical] = [round(float(x), 6) for x in mean]

        merged_speakers = {s["speaker"] for s in segments}
        total_speech_sec = sum(s["end"] - s["start"] for s in segments)

        result = {
            "segments": segments,
            "duration_sec": round(duration_sec, 3),
            "overlap_sec": round(overlap_sec, 3),
            "num_speakers_raw": len(raw_speakers),
            "num_speakers_merged": len(merged_speakers),
            "gpu_name": _gpu_name(),
            "diagnostics": {
                "download_time_sec": round(download_elapsed, 1),
                "decode_time_sec": round(convert_elapsed, 1),
                "diarization_time_sec": round(diarize_elapsed, 1),
                "embedding_time_sec": round(embed_elapsed, 1),
                "audio_size_mb": round(bytes_written / 1e6, 1),
                "num_segments": len(segments),
                "total_speech_sec": round(total_speech_sec, 1),
                "recluster_threshold": recluster_threshold,
                "recluster_remap": remap,
            },
        }
        if return_embeddings and canonical_centroids:
            result["embeddings"] = canonical_centroids

        return result

    except Exception as exc:
        log.exception("[job:%s] Diarization failed", job_id)
        return {"error": str(exc)}

    finally:
        for p in (tmp_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


runpod.serverless.start({"handler": handler})
