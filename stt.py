"""
stt.py — Sarvam AI Speech-to-Text logic

Three transport paths, all using model="saaras:v3":
  REST API  : sync, single file ≤30 s, all 5 modes
  Batch API : async job, up to 1 hour, uses high-level SpeechToTextJob object
  Streaming : async WebSocket via AsyncSarvamAI (WAV / PCM only)

Modes: transcribe | translate | verbatim | translit | codemix
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import os
import tempfile
from typing import Callable, Optional

from sarvamai import AsyncSarvamAI, SarvamAI
from sarvamai.core.api_error import ApiError

# ── Constants ────────────────────────────────────────────────────────────────

STT_MODEL = "saaras:v3"

LANGUAGES = [
    "unknown", "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN",
    "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
    "as-IN", "ur-IN", "ne-IN",
]

MODES = ["transcribe", "translate", "verbatim", "translit", "codemix"]


# ── Client helpers ────────────────────────────────────────────────────────────

def make_client(api_key: str) -> SarvamAI:
    return SarvamAI(api_subscription_key=api_key)


def make_async_client(api_key: str) -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=api_key)


def _api_error_msg(e: ApiError) -> str:
    """Extract a human-readable message from ApiError."""
    code = e.status_code or 0
    body = e.body or {}
    if isinstance(body, dict):
        inner = body.get("error", body)
        detail = inner.get("message", str(inner)) if isinstance(inner, dict) else str(inner)
    else:
        detail = str(body)

    labels = {
        400: "Bad request",
        403: "Invalid or missing API key (403)",
        422: "Unprocessable entity — bad audio format or params (422)",
        429: "Rate limit / quota exceeded — wait and retry (429)",
        500: "Sarvam server error — retry or contact support (500)",
        503: "Service temporarily overloaded — retry with backoff (503)",
    }
    return f"{labels.get(code, f'HTTP {code}')}: {detail}"


# ── REST API ──────────────────────────────────────────────────────────────────

def stt_rest(
    client: SarvamAI,
    audio_bytes: bytes,
    filename: str,
    mode: str,
    language_code: str,
) -> dict:
    """
    Synchronous single-file transcription (≤30 s audio).

    Returns {"transcript", "language_code", "request_id", "raw"} or {"error"}.
    """
    try:
        resp = client.speech_to_text.transcribe(
            file=(filename, io.BytesIO(audio_bytes)),
            model=STT_MODEL,
            mode=mode,
            language_code=language_code,
        )
        return {
            "transcript": resp.transcript or "",
            "language_code": resp.language_code,
            "request_id": resp.request_id,
            "raw": resp.model_dump(),
        }
    except ApiError as e:
        return {"error": _api_error_msg(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# ── Batch API ─────────────────────────────────────────────────────────────────

def stt_batch(
    client: SarvamAI,
    audio_bytes: bytes,
    filename: str,
    mode: str,
    language_code: str,
    with_diarization: bool = False,
    num_speakers: Optional[int] = None,
    poll_interval: int = 5,
    timeout: int = 600,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Long-audio batch transcription using the SpeechToTextJob high-level API.

    Flow:
      create_job(model, mode, language_code, ...)
      → upload_files([tmp_path])     # SDK uploads bytes to S3 signed URL
      → start()
      → wait_until_complete()
      → download_outputs(output_dir)
      → get_file_results()

    audio_bytes is written to a named temp file because upload_files() takes paths.

    Returns {"transcript", "file_results", "output_dir"} or {"error"}.
    """

    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    ext = os.path.splitext(filename)[1] or ".wav"
    tmp_path: Optional[str] = None

    try:
        # Write audio to temp file — upload_files() requires real paths
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        output_dir = tempfile.mkdtemp(prefix="sarvam_batch_out_")

        log("Creating batch job…")
        job_kwargs: dict = dict(model=STT_MODEL, mode=mode)
        if language_code and language_code != "unknown":
            job_kwargs["language_code"] = language_code
        if with_diarization:
            job_kwargs["with_diarization"] = True
            if num_speakers:
                job_kwargs["num_speakers"] = num_speakers

        job = client.speech_to_text_job.create_job(**job_kwargs)
        log(f"Job created: {job.job_id}")

        log("Uploading audio to signed URL…")
        if not job.upload_files(file_paths=[tmp_path]):
            return {"error": "File upload to signed URL failed."}

        log("Starting job…")
        job.start()

        log(f"Polling for completion (interval={poll_interval}s, timeout={timeout}s)…")
        job.wait_until_complete(poll_interval=poll_interval, timeout=timeout)

        if job.is_failed():
            status = job.get_status()
            return {"error": f"Batch job failed with state: {status.state}"}

        log("Downloading outputs…")
        job.download_outputs(output_dir=output_dir)

        file_results = job.get_file_results()
        transcript = _read_batch_transcript(output_dir)

        return {
            "transcript": transcript,
            "file_results": file_results,
            "output_dir": output_dir,
        }

    except ApiError as e:
        return {"error": _api_error_msg(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _read_batch_transcript(output_dir: str) -> str:
    """Pull transcript text from the first JSON output file in output_dir."""
    for fname in os.listdir(output_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(output_dir, fname)) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return str(data)
        if "transcript" in data:
            return data["transcript"]
        entries = data.get("diarized_transcript", {}).get("entries", [])
        if entries:
            return " ".join(e.get("transcript", "") for e in entries)
        return str(data)
    return "(no output file found)"


# ── Streaming API ─────────────────────────────────────────────────────────────

def stt_streaming_sync(
    api_key: str,
    audio_bytes: bytes,
    language_code: str,
    mode: str,
    sample_rate: int = 16000,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Blocking wrapper around the async WebSocket streaming implementation.

    audio_bytes must be WAV or raw PCM — streaming only supports those two formats.
    Sends the entire file as one base64 chunk, then flushes to force immediate processing.

    Returns {"transcript", "segments"} or {"error"}.
    """
    coro = _stt_streaming_async(api_key, audio_bytes, language_code, mode, sample_rate, progress_cb)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Already inside a running event loop (e.g., Streamlit's async context on some runners)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


async def _stt_streaming_async(
    api_key: str,
    audio_bytes: bytes,
    language_code: str,
    mode: str,
    sample_rate: int,
    progress_cb: Optional[Callable[[str], None]],
) -> dict:
    """
    Async core: AsyncSarvamAI + async-with + async-for.

    Docs pattern:
        async with client.speech_to_text_streaming.connect(...) as ws:
            await ws.transcribe(audio=b64, encoding="audio/wav", sample_rate=16000)
            await ws.flush()
            async for message in ws:
                ...
    """

    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    segments: list[str] = []
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    try:
        client = make_async_client(api_key)

        async with client.speech_to_text_streaming.connect(
            model=STT_MODEL,
            mode=mode,
            language_code=language_code,
            high_vad_sensitivity="true",
            flush_signal="true",
            input_audio_codec="wav",
            sample_rate=str(sample_rate),
        ) as ws:
            log("WebSocket connected. Sending audio…")
            await ws.transcribe(audio=audio_b64, encoding="audio/wav", sample_rate=sample_rate)
            await ws.flush()
            log("Audio flushed. Awaiting transcript…")

            async for message in ws:
                if message is None:
                    break
                # SDK may yield dicts or SpeechToTextStreamingResponse objects
                if isinstance(message, dict):
                    msg_type = message.get("type", "")
                    if msg_type in ("speech_start", "speech_end"):
                        log(f"VAD signal: {msg_type}")
                    else:
                        text = message.get("text") or message.get("transcript", "")
                        if text:
                            segments.append(text)
                            log(f"Segment: {text}")
                else:
                    text = getattr(message, "transcript", None)
                    if text:
                        segments.append(text)
                        log(f"Segment: {text}")

        return {"transcript": " ".join(segments), "segments": segments}

    except ApiError as e:
        return {"error": _api_error_msg(e)}
    except ConnectionError as e:
        return {"error": f"WebSocket connection error: {e}"}
    except Exception as e:
        return {"error": f"Streaming error: {e}"}
