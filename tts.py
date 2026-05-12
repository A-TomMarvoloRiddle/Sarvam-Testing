"""
REST + HTTP Stream: unchanged (batch/file-based).
WebSocket: real-time streaming with progressive audio chunk delivery.

Real-time TTS WebSocket architecture:
  Caller opens a TtsStreamingSession (holds one persistent WS connection).
  Caller calls send_text(sentence) to stream text chunks.
  Background async recv loop collects AudioOutput chunks into result_queue.
  Caller drains result_queue to get decoded PCM/MP3 bytes progressively.
  UI plays each chunk immediately using Web Audio API (via custom HTML component).
"""

import asyncio
import base64
import concurrent.futures
import io
import queue
import threading
from typing import Callable, Iterator, List, Optional

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
TTS_MODELS    = ["bulbul:v3", "bulbul:v2"]
REST_CODECS   = ["wav", "mp3", "flac", "opus", "aac", "linear16", "mulaw", "alaw"]
STREAM_CODECS = ["mp3", "wav", "aac", "opus", "flac", "linear16", "mulaw", "alaw"]
WS_CODECS     = ["mp3", "wav", "aac"]
BITRATES      = ["32k", "64k", "96k", "128k", "192k"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_client(api_key: str) -> SarvamAI:
    return SarvamAI(api_subscription_key=api_key)

def make_async_client(api_key: str) -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=api_key)

def _api_err(e: ApiError) -> str:
    code  = e.status_code or 0
    body  = e.body or {}
    inner = body.get("error", body) if isinstance(body, dict) else {}
    msg   = inner.get("message", str(body)) if isinstance(inner, dict) else str(inner)
    labels = {400: "Bad request", 403: "Invalid API key",
              422: "Bad text/params", 429: "Rate limit", 500: "Server error"}
    return f"{labels.get(code, f'HTTP {code}')}: {msg}"

def _decode_audio(response) -> bytes:
    raw = response.audios[0] if response.audios else None
    if raw is None: raise ValueError("No audio in response.")
    return base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)


# ── REST ──────────────────────────────────────────────────────────────────────

def tts_rest(client, text, target_language_code,
             speaker="shubh", model="bulbul:v3", output_audio_codec="wav",
             pitch=None, loudness=None, pace=None, temperature=None,
             speech_sample_rate=None, enable_preprocessing=None,
             enable_cached_responses=None) -> dict:
    kw = dict(text=text, target_language_code=target_language_code,
               speaker=speaker, model=model, output_audio_codec=output_audio_codec)
    if pace is not None: kw["pace"] = pace
    if model == "bulbul:v2":
        if pitch    is not None: kw["pitch"]    = pitch
        if loudness is not None: kw["loudness"] = loudness
    if model == "bulbul:v3":
        if temperature         is not None: kw["temperature"]         = temperature
        if speech_sample_rate  is not None: kw["speech_sample_rate"]  = speech_sample_rate
        if enable_preprocessing is not None: kw["enable_preprocessing"] = enable_preprocessing
    if enable_cached_responses is not None:
        kw["enable_cached_responses"] = enable_cached_responses
    try:
        r = client.text_to_speech.convert(**kw)
        return {"audio_bytes": _decode_audio(r), "codec": output_audio_codec,
                "request_id": r.request_id, "raw": r.model_dump()}
    except ApiError as e: return {"error": _api_err(e)}
    except Exception as e: return {"error": str(e)}


# ── HTTP Stream ───────────────────────────────────────────────────────────────

def tts_http_stream(client, text, target_language_code,
                    speaker="shubh", model="bulbul:v3",
                    output_audio_codec="mp3", output_audio_bitrate="128k",
                    pace=None, temperature=None, speech_sample_rate=None,
                    enable_preprocessing=None, progress_cb=None) -> dict:
    kw = dict(text=text, target_language_code=target_language_code,
               speaker=speaker, model=model,
               output_audio_codec=output_audio_codec,
               output_audio_bitrate=output_audio_bitrate)
    if pace is not None: kw["pace"] = pace
    if model == "bulbul:v3":
        if temperature         is not None: kw["temperature"]         = temperature
        if speech_sample_rate  is not None: kw["speech_sample_rate"]  = speech_sample_rate
        if enable_preprocessing is not None: kw["enable_preprocessing"] = enable_preprocessing
    try:
        buf, n = io.BytesIO(), 0
        for chunk in client.text_to_speech.convert_stream(**kw):
            buf.write(chunk); n += 1
            if progress_cb: progress_cb(buf.tell())
        audio = buf.getvalue()
        if not audio: return {"error": "Stream returned no audio."}
        return {"audio_bytes": audio, "codec": output_audio_codec, "chunks_received": n}
    except ApiError as e: return {"error": _api_err(e)}
    except Exception as e: return {"error": str(e)}


# ── Real-time WebSocket TTS ───────────────────────────────────────────────────
#
# TtsStreamingSession keeps one AsyncSarvamAI WS alive across multiple
# send_text() calls — exactly the conversational agent use-case.
#
# Usage:
#   session = TtsStreamingSession(api_key, language_code, speaker, ...)
#   session.start()
#   session.send_text("Hello, how can I help you?")
#   for chunk_bytes in session.iter_audio():   # yields as chunks arrive
#       play(chunk_bytes)
#   session.stop()


class TtsStreamingSession:
    """
    Persistent TTS WebSocket session.

    • One background thread + asyncio loop.
    • Text is queued via send_text() from any thread.
    • Decoded audio chunks are available via drain_audio() (non-blocking).
    • iter_audio() is a blocking generator that yields chunks until done.
    """

    def __init__(self, api_key: str, target_language_code: str,
                 speaker: str = "shubh", model: str = "bulbul:v3",
                 output_audio_codec: str = "mp3",
                 output_audio_bitrate: str = "128k",
                 pace: float = 1.0, pitch: float = 0.0,
                 loudness: float = 1.0, speech_sample_rate: int = 22050,
                 enable_preprocessing: bool = False):
        self.api_key               = api_key
        self.target_language_code  = target_language_code
        self.speaker               = speaker
        self.model                 = model
        self.output_audio_codec    = output_audio_codec
        self.output_audio_bitrate  = output_audio_bitrate
        self.pace                  = pace
        self.pitch                 = pitch
        self.loudness              = loudness
        self.speech_sample_rate    = speech_sample_rate
        self.enable_preprocessing  = enable_preprocessing

        self._text_queue:  queue.Queue[Optional[str]] = queue.Queue()
        self._audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.error: Optional[str] = None

    # ── public ───────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="sarvam-tts-ws")
        self._thread.start()

    def send_text(self, text: str):
        """Queue a text chunk to be synthesized. Thread-safe."""
        if self._running:
            self._text_queue.put(text)

    def flush(self):
        """Signal end-of-turn: flush the WS buffer."""
        self._text_queue.put("")  # empty string sentinel = flush only

    def stop(self):
        """Gracefully stop: drain remaining audio then close."""
        self._running = False
        self._text_queue.put(None)  # poison pill
        if self._thread: self._thread.join(timeout=8)

    def drain_audio(self) -> List[bytes]:
        """Non-blocking: return all available audio chunks."""
        out = []
        while True:
            try:
                chunk = self._audio_queue.get_nowait()
                if chunk is None: break   # end-of-stream sentinel
                out.append(chunk)
            except queue.Empty:
                break
        return out

    def iter_audio(self, timeout: float = 30.0) -> Iterator[bytes]:
        """
        Blocking generator: yields decoded audio bytes as chunks arrive.
        Ends when end-of-stream sentinel (None) is received or timeout exceeded.
        """
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None: return
                yield chunk
                deadline = time.monotonic() + timeout  # reset on activity
            except queue.Empty:
                if not self._running and self._thread and not self._thread.is_alive():
                    return

    # ── background thread ────────────────────────────────────────────────────

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        finally:
            loop.close()

    async def _ws_loop(self):
        client = make_async_client(self.api_key)
        try:
            async with client.text_to_speech_streaming.connect(
                model=self.model, send_completion_event="true",
            ) as ws:
                await ws.configure(
                    target_language_code = self.target_language_code,
                    speaker              = self.speaker,
                    pitch                = self.pitch,
                    pace                 = self.pace,
                    loudness             = self.loudness,
                    speech_sample_rate   = self.speech_sample_rate,
                    enable_preprocessing = self.enable_preprocessing,
                    output_audio_codec   = self.output_audio_codec,
                    output_audio_bitrate = self.output_audio_bitrate,
                )

                send_task = asyncio.create_task(self._send_loop(ws))
                recv_task = asyncio.create_task(self._recv_loop(ws))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)
        except ApiError as e:
            self.error = _api_err(e)
        except Exception as e:
            self.error = str(e)
        finally:
            self._audio_queue.put(None)  # end-of-stream sentinel

    async def _send_loop(self, ws):
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                text = await loop.run_in_executor(
                    None, lambda: self._text_queue.get(timeout=0.1)
                )
            except queue.Empty:
                continue

            if text is None:        # poison pill → stop
                break
            elif text == "":        # flush sentinel
                await ws.flush()
            else:
                await ws.convert(text)
                await ws.flush()    # flush after each sentence for low latency

    async def _recv_loop(self, ws):
        async for message in ws:
            if isinstance(message, AudioOutput):
                chunk = base64.b64decode(message.data.audio)
                self._audio_queue.put(chunk)
            elif isinstance(message, EventResponse):
                evt = str(message.data.event_type) if message.data else ""
                if evt == "final" and not self._running:
                    break
            if not self._running:
                break


# ── Convenience: one-shot blocking WS call (used by "WebSocket" UI tab) ───────

def tts_websocket_sync(api_key, text, target_language_code,
                       speaker="shubh", model="bulbul:v3",
                       output_audio_codec="mp3", output_audio_bitrate="128k",
                       pace=1.0, pitch=0.0, loudness=1.0,
                       speech_sample_rate=22050, enable_preprocessing=False,
                       progress_cb=None) -> dict:
    """
    One-shot: send full text, collect all audio, return when done.
    Reuses TtsStreamingSession internally.
    """
    session = TtsStreamingSession(
        api_key=api_key, target_language_code=target_language_code,
        speaker=speaker, model=model,
        output_audio_codec=output_audio_codec, output_audio_bitrate=output_audio_bitrate,
        pace=pace, pitch=pitch, loudness=loudness,
        speech_sample_rate=speech_sample_rate, enable_preprocessing=enable_preprocessing,
    )

    def _log(m):
        if progress_cb: progress_cb(m)

    session.start()
    _log("WebSocket connected. Sending text…")
    session.send_text(text)

    chunks: List[bytes] = []
    for chunk in session.iter_audio(timeout=30.0):
        chunks.append(chunk)
        _log(f"Chunk received: {len(chunk):,} bytes")

    session.stop()

    if session.error:
        return {"error": session.error}
    audio = b"".join(chunks)
    if not audio:
        return {"error": "No audio received."}
    return {"audio_bytes": audio, "codec": output_audio_codec,
            "chunks_received": len(chunks)}
