"""
tts.py — Sarvam AI Text-to-Speech logic

Five controls:
  REST Basic        : text → base64 audio, minimal params
  REST Voice        : adds full speaker selection
  REST Advanced     : model-specific advanced params (v2 vs v3)
  HTTP Stream       : convert_stream() → raw binary chunks
  WebSocket Stream  : AsyncSarvamAI, configure → convert → flush → async-for AudioOutput
"""

import asyncio
import base64
import concurrent.futures
import io
from typing import Callable, Optional

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse, SarvamAI
from sarvamai.core.api_error import ApiError

# ── Constants ─────────────────────────────────────────────────────────────────

TTS_LANGUAGES = [
    "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN",
    "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
]

TTS_SPEAKERS = [
    "anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh",
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan",
    "sumit", "roopa", "kabir", "aayan", "shubh", "ashutosh", "advait",
    "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti",
    "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
]

TTS_MODELS = ["bulbul:v3", "bulbul:v2"]

REST_CODECS  = ["wav", "mp3", "flac", "opus", "aac", "linear16", "mulaw", "alaw"]
STREAM_CODECS = ["mp3", "wav", "aac", "opus", "flac", "linear16", "mulaw", "alaw"]
WS_CODECS     = ["mp3", "wav", "aac"]    # WebSocket supports fewer codecs
BITRATES      = ["32k", "64k", "96k", "128k", "192k"]

# ── Client helpers ─────────────────────────────────────────────────────────────

def make_client(api_key: str) -> SarvamAI:
    return SarvamAI(api_subscription_key=api_key)


def make_async_client(api_key: str) -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=api_key)


def _api_error_msg(e: ApiError) -> str:
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
        422: "Unprocessable entity — text too long or bad params (422)",
        429: "Rate limit / quota exceeded (429)",
        500: "Sarvam server error (500)",
    }
    return f"{labels.get(code, f'HTTP {code}')}: {detail}"


def _decode_audio(response) -> bytes:
    """Decode base64 audio from REST response.audios[0]."""
    raw = response.audios[0] if response.audios else None
    if raw is None:
        raise ValueError("No audio data in response.")
    return base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)


# ── REST API ──────────────────────────────────────────────────────────────────

def tts_rest(
    client: SarvamAI,
    text: str,
    target_language_code: str,
    speaker: str = "shubh",
    model: str = "bulbul:v3",
    output_audio_codec: str = "wav",
    # bulbul:v2 params
    pitch: Optional[float] = None,
    loudness: Optional[float] = None,
    # shared pace
    pace: Optional[float] = None,
    # bulbul:v3 params
    temperature: Optional[float] = None,
    speech_sample_rate: Optional[int] = None,
    enable_preprocessing: Optional[bool] = None,
    enable_cached_responses: Optional[bool] = None,
) -> dict:
    """
    REST synthesis. Returns {"audio_bytes", "codec", "request_id", "raw"} or {"error"}.

    Model-specific advanced params:
      bulbul:v2 → pitch, pace, loudness
      bulbul:v3 → pace, temperature, speech_sample_rate, enable_preprocessing
    """
    kwargs: dict = dict(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        model=model,
        output_audio_codec=output_audio_codec,
    )
    # Only pass params that are supported by / relevant to the chosen model
    if pace is not None:
        kwargs["pace"] = pace
    if model == "bulbul:v2":
        if pitch is not None:
            kwargs["pitch"] = pitch
        if loudness is not None:
            kwargs["loudness"] = loudness
    if model == "bulbul:v3":
        if temperature is not None:
            kwargs["temperature"] = temperature
        if speech_sample_rate is not None:
            kwargs["speech_sample_rate"] = speech_sample_rate
        if enable_preprocessing is not None:
            kwargs["enable_preprocessing"] = enable_preprocessing
    if enable_cached_responses is not None:
        kwargs["enable_cached_responses"] = enable_cached_responses

    try:
        resp = client.text_to_speech.convert(**kwargs)
        audio = _decode_audio(resp)
        return {
            "audio_bytes": audio,
            "codec": output_audio_codec,
            "request_id": resp.request_id,
            "raw": resp.model_dump(),
        }
    except ApiError as e:
        return {"error": _api_error_msg(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# ── HTTP Streaming ─────────────────────────────────────────────────────────────

def tts_http_stream(
    client: SarvamAI,
    text: str,
    target_language_code: str,
    speaker: str = "shubh",
    model: str = "bulbul:v3",
    output_audio_codec: str = "mp3",
    output_audio_bitrate: str = "128k",
    pace: Optional[float] = None,
    temperature: Optional[float] = None,
    speech_sample_rate: Optional[int] = None,
    enable_preprocessing: Optional[bool] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> dict:
    """
    HTTP chunked streaming: convert_stream() returns Iterator[bytes] — raw binary, no decode.

    Returns {"audio_bytes", "codec", "chunks_received"} or {"error"}.
    """
    kwargs: dict = dict(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        model=model,
        output_audio_codec=output_audio_codec,
        output_audio_bitrate=output_audio_bitrate,
    )
    if pace is not None:
        kwargs["pace"] = pace
    if model == "bulbul:v3":
        if temperature is not None:
            kwargs["temperature"] = temperature
        if speech_sample_rate is not None:
            kwargs["speech_sample_rate"] = speech_sample_rate
        if enable_preprocessing is not None:
            kwargs["enable_preprocessing"] = enable_preprocessing

    try:
        buf = io.BytesIO()
        chunks_received = 0
        for chunk in client.text_to_speech.convert_stream(**kwargs):
            buf.write(chunk)
            chunks_received += 1
            if progress_cb:
                progress_cb(buf.tell())

        audio = buf.getvalue()
        if not audio:
            return {"error": "Stream returned no audio data."}
        return {"audio_bytes": audio, "codec": output_audio_codec, "chunks_received": chunks_received}

    except ApiError as e:
        return {"error": _api_error_msg(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


# ── WebSocket Streaming ────────────────────────────────────────────────────────

def tts_websocket_sync(
    api_key: str,
    text: str,
    target_language_code: str,
    speaker: str = "shubh",
    model: str = "bulbul:v3",
    output_audio_codec: str = "mp3",
    output_audio_bitrate: str = "128k",
    pace: float = 1.0,
    pitch: float = 0.0,
    loudness: float = 1.0,
    speech_sample_rate: int = 22050,
    enable_preprocessing: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Blocking wrapper for async WebSocket TTS.
    Returns {"audio_bytes", "codec", "chunks_received", "events"} or {"error"}.
    """
    coro = _tts_websocket_async(
        api_key, text, target_language_code, speaker, model,
        output_audio_codec, output_audio_bitrate,
        pace, pitch, loudness, speech_sample_rate, enable_preprocessing, progress_cb,
    )
    try:
        return asyncio.run(coro)
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


async def _tts_websocket_async(
    api_key: str,
    text: str,
    target_language_code: str,
    speaker: str,
    model: str,
    output_audio_codec: str,
    output_audio_bitrate: str,
    pace: float,
    pitch: float,
    loudness: float,
    speech_sample_rate: int,
    enable_preprocessing: bool,
    progress_cb: Optional[Callable[[str], None]],
) -> dict:
    """
    Docs pattern (AsyncSarvamAI):
        async with client.text_to_speech_streaming.connect(model=..., send_completion_event="true") as ws:
            await ws.configure(target_language_code=..., speaker=...)
            await ws.convert(text)
            await ws.flush()
            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk = base64.b64decode(message.data.audio)
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final": break
    """

    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    audio_chunks: list[bytes] = []
    events: list[str] = []

    try:
        client = make_async_client(api_key)

        async with client.text_to_speech_streaming.connect(
            model=model,
            send_completion_event="true",
        ) as ws:
            log("WebSocket connected. Sending config…")
            await ws.configure(
                target_language_code=target_language_code,
                speaker=speaker,
                pitch=pitch,
                pace=pace,
                loudness=loudness,
                speech_sample_rate=speech_sample_rate,
                enable_preprocessing=enable_preprocessing,
                output_audio_codec=output_audio_codec,
                output_audio_bitrate=output_audio_bitrate,
            )

            log("Sending text…")
            await ws.convert(text)
            await ws.flush()
            log("Awaiting audio chunks…")

            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk = base64.b64decode(message.data.audio)
                    audio_chunks.append(chunk)
                    log(f"Audio chunk: {len(chunk):,} bytes")
                elif isinstance(message, EventResponse):
                    evt = message.data.event_type if message.data else "unknown"
                    events.append(str(evt))
                    log(f"Event: {evt}")
                    if str(evt) == "final":
                        break

        audio_bytes = b"".join(audio_chunks)
        if not audio_bytes:
            return {"error": "No audio received via WebSocket.", "events": events}
        return {
            "audio_bytes": audio_bytes,
            "codec": output_audio_codec,
            "chunks_received": len(audio_chunks),
            "events": events,
        }

    except ApiError as e:
        return {"error": _api_error_msg(e)}
    except ConnectionError as e:
        return {"error": f"WebSocket connection error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
