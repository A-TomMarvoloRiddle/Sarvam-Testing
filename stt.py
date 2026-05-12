"""
REST + Batch: unchanged (file-based).
Streaming: real-time mic via streamlit-webrtc.

Real-time streaming architecture:
  webrtc AudioProcessor.recv_queued()
    → puts raw PCM frames into a thread-safe audio_queue
  Background daemon thread (one per Streamlit session)
    → runs its own asyncio event loop
    → holds one persistent Sarvam STT WebSocket open
    → reads audio_queue, sends base64 PCM chunks via ws.transcribe()
    → on VAD silence / flush → ws.flush() triggers transcript
    → puts transcript strings into transcript_queue
  Streamlit UI (polling via st_autorefresh every 500 ms)
    → drains transcript_queue into session_state
    → rerenders transcript display
"""

import asyncio
import base64
import io
import json
import os
import queue
import struct
import tempfile
import threading
from typing import Callable, List, Optional

import numpy as np
from sarvamai import AsyncSarvamAI, SarvamAI
from sarvamai.core.api_error import ApiError

# ── Constants ─────────────────────────────────────────────────────────────────

STT_MODEL   = "saaras:v3"
SAMPLE_RATE = 16000          # Sarvam streaming requires 16 kHz
CHUNK_MS    = 100            # ms of audio per WS send (100 ms = 1600 samples @ 16kHz)

LANGUAGES = [
    "unknown", "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN",
    "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
    "as-IN", "ur-IN", "ne-IN",
]
MODES = ["transcribe", "translate", "verbatim", "translit", "codemix"]


# ── Client factories ──────────────────────────────────────────────────────────

def make_client(api_key: str) -> SarvamAI:
    return SarvamAI(api_subscription_key=api_key)

def make_async_client(api_key: str) -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=api_key)


# ── Error helper ──────────────────────────────────────────────────────────────

def _api_err(e: ApiError) -> str:
    code  = e.status_code or 0
    body  = e.body or {}
    inner = body.get("error", body) if isinstance(body, dict) else {}
    msg   = inner.get("message", str(body)) if isinstance(inner, dict) else str(inner)
    tags  = {
        400: "Bad request",           403: "Invalid API key",
        422: "Bad format/params",     429: "Rate limit exceeded — wait and retry",
        500: "Sarvam server error",   503: "Service overloaded — retry with backoff",
    }
    return f"{tags.get(code, f'HTTP {code}')}: {msg}"


# ── REST ──────────────────────────────────────────────────────────────────────

def stt_rest(client, audio_bytes, filename, mode, language_code) -> dict:
    """Sync single-file transcription (≤30 s)."""
    try:
        r = client.speech_to_text.transcribe(
            file=(filename, io.BytesIO(audio_bytes)),
            model=STT_MODEL, mode=mode, language_code=language_code,
        )
        return {"transcript": r.transcript or "", "language_code": r.language_code,
                "request_id": r.request_id, "raw": r.model_dump()}
    except ApiError as e:
        return {"error": _api_err(e)}
    except Exception as e:
        return {"error": str(e)}


# ── Batch ─────────────────────────────────────────────────────────────────────

def stt_batch(client, audio_bytes, filename, mode, language_code,
              with_diarization=False, num_speakers=None,
              poll_interval=5, timeout=600, progress_cb=None) -> dict:
    """Job-based long-audio transcription."""
    log = lambda m: progress_cb(m) if progress_cb else None
    ext = os.path.splitext(filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes); tmp_path = f.name
        out_dir = tempfile.mkdtemp(prefix="sarvam_batch_")

        log("Creating job…")
        kw = dict(model=STT_MODEL, mode=mode)
        if language_code and language_code != "unknown":
            kw["language_code"] = language_code
        if with_diarization:
            kw["with_diarization"] = True
            if num_speakers: kw["num_speakers"] = num_speakers

        job = client.speech_to_text_job.create_job(**kw)
        log(f"Job: {job.job_id}")
        if not job.upload_files(file_paths=[tmp_path]):
            return {"error": "Upload to signed URL failed."}
        log("Starting…"); job.start()
        log("Polling…"); job.wait_until_complete(poll_interval=poll_interval, timeout=timeout)
        if job.is_failed():
            return {"error": f"Job failed: {job.get_status().state}"}
        log("Downloading…"); job.download_outputs(output_dir=out_dir)
        results = job.get_file_results()
        return {"transcript": _read_batch_transcript(out_dir),
                "file_results": results, "output_dir": out_dir}
    except ApiError as e:
        return {"error": _api_err(e)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass


def _read_batch_transcript(out_dir: str) -> str:
    for fn in os.listdir(out_dir):
        if not fn.endswith(".json"): continue
        with open(os.path.join(out_dir, fn)) as f:
            d = json.load(f)
        if not isinstance(d, dict): return str(d)
        if "transcript" in d: return d["transcript"]
        entries = d.get("diarized_transcript", {}).get("entries", [])
        if entries: return " ".join(e.get("transcript", "") for e in entries)
        return str(d)
    return "(no output)"


# ── Real-time Streaming ───────────────────────────────────────────────────────
#
# StreamingSession manages one persistent Sarvam STT WebSocket connection in
# a background daemon thread.  It exposes two thread-safe queues:
#
#   audio_queue   : caller pushes raw PCM bytes (s16le, 16 kHz, mono)
#   result_queue  : session pushes {"transcript": str} or {"error": str}
#
# The webrtc AudioProcessor calls push_audio() from the webrtc event loop.
# Streamlit UI polls result_queue via drain_results().


class StreamingSession:
    """
    Owns one background thread + asyncio loop + Sarvam STT WebSocket.

    Lifecycle:
        session = StreamingSession(api_key, language_code, mode)
        session.start()          # launches background thread + connects WS
        session.push_audio(pcm)  # feed raw PCM bytes (s16le, 16 kHz, mono)
        session.stop()           # closes WS, joins thread
        segments = session.drain_results()  # list of transcript strings
    """

    # How many bytes to accumulate before sending a WS frame.
    # 100 ms @ 16 kHz s16le = 3200 bytes
    _CHUNK_BYTES = int(SAMPLE_RATE * 2 * CHUNK_MS / 1000)

    def __init__(self, api_key: str, language_code: str, mode: str):
        self.api_key        = api_key
        self.language_code  = language_code
        self.mode           = mode

        self._audio_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=200)
        self._result_queue: queue.Queue[dict]           = queue.Queue()
        self._pcm_buf       = bytearray()
        self._thread: Optional[threading.Thread]        = None
        self._loop:  Optional[asyncio.AbstractEventLoop] = None
        self._running       = False
        self.error: Optional[str] = None

    # ── public API ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="sarvam-stt-ws")
        self._thread.start()

    def push_audio(self, pcm_bytes: bytes):
        """Feed raw PCM (s16le, 16 kHz, mono) from any thread."""
        if not self._running: return
        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass  # drop if queue is full (backpressure)

    def flush(self):
        """Signal end-of-utterance: send None sentinel to trigger WS flush."""
        try:
            self._audio_queue.put_nowait(None)  # flush sentinel
        except queue.Full:
            pass

    def stop(self):
        self._running = False
        try: self._audio_queue.put_nowait(None)
        except queue.Full: pass
        if self._thread: self._thread.join(timeout=5)

    def drain_results(self) -> List[dict]:
        """Non-blocking drain of all pending results. Returns list of dicts."""
        out = []
        while True:
            try: out.append(self._result_queue.get_nowait())
            except queue.Empty: break
        return out

    # ── background thread ───────────────────────────────────────────────────

    def _run(self):
        """Entry point for daemon thread — creates its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        finally:
            self._loop.close()

    async def _ws_loop(self):
        """
        Persistent async loop:
          1. Connect Sarvam STT WebSocket.
          2. Concurrently: send audio chunks AND receive transcripts.
          3. On stop signal, flush + close gracefully.
        """
        client = make_async_client(self.api_key)
        try:
            async with client.speech_to_text_streaming.connect(
                model           = STT_MODEL,
                mode            = self.mode,
                language_code   = self.language_code,
                high_vad_sensitivity = "true",
                flush_signal    = "true",
                input_audio_codec = "pcm_s16le",
                sample_rate     = str(SAMPLE_RATE),
            ) as ws:
                send_task = asyncio.create_task(self._send_loop(ws))
                recv_task = asyncio.create_task(self._recv_loop(ws))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)
        except ApiError as e:
            self._result_queue.put({"error": _api_err(e)})
        except Exception as e:
            self._result_queue.put({"error": f"WS error: {e}"})

    async def _send_loop(self, ws):
        """Read PCM from queue, accumulate into chunks, send as b64 PCM."""
        buf = bytearray()
        while self._running:
            try:
                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._audio_queue.get(timeout=0.1)
                )
            except queue.Empty:
                continue

            if pcm is None:
                # Flush sentinel: send whatever is buffered then flush WS
                if buf:
                    await self._send_chunk(ws, bytes(buf))
                    buf.clear()
                await ws.flush()
                if not self._running:
                    break
                continue

            buf.extend(pcm)
            # Send in fixed-size chunks for smooth streaming
            while len(buf) >= self._CHUNK_BYTES:
                await self._send_chunk(ws, bytes(buf[:self._CHUNK_BYTES]))
                buf = buf[self._CHUNK_BYTES:]

        # Drain remainder on exit
        if buf:
            await self._send_chunk(ws, bytes(buf))
        await ws.flush()

    async def _send_chunk(self, ws, pcm: bytes):
        b64 = base64.b64encode(pcm).decode()
        await ws.transcribe(audio=b64, encoding="audio/x-raw", sample_rate=SAMPLE_RATE)

    async def _recv_loop(self, ws):
        """Read transcripts from WS and put into result_queue."""
        async for message in ws:
            if message is None:
                break
            if isinstance(message, dict):
                msg_type = message.get("type", "")
                if msg_type in ("speech_start", "speech_end"):
                    self._result_queue.put({"signal": msg_type})
                else:
                    text = message.get("text") or message.get("transcript", "")
                    if text:
                        self._result_queue.put({"transcript": text})
            else:
                text = getattr(message, "transcript", None)
                if text:
                    self._result_queue.put({"transcript": text})
            if not self._running:
                break


# ── webrtc AudioProcessor ─────────────────────────────────────────────────────

def make_audio_processor(session: "StreamingSession"):
    """
    Factory that returns an AudioProcessorBase subclass wired to the given session.

    Called by webrtc_streamer(audio_processor_factory=...).
    recv_queued() is called by aiortc's event loop per batch of frames.
    """
    from streamlit_webrtc import AudioProcessorBase
    import av as _av

    class _SarvamAudioProcessor(AudioProcessorBase):
        def recv(self, frame: _av.AudioFrame) -> _av.AudioFrame:
            pcm = _frame_to_pcm16_mono(frame)
            session.push_audio(pcm)
            return frame  # pass through (sendback_audio=False avoids echo)

        async def recv_queued(self, frames: List[_av.AudioFrame]) -> List[_av.AudioFrame]:
            for frame in frames:
                pcm = _frame_to_pcm16_mono(frame)
                session.push_audio(pcm)
            return frames

        def on_ended(self):
            session.flush()

    return _SarvamAudioProcessor


def _frame_to_pcm16_mono(frame) -> bytes:
    """
    Convert av.AudioFrame (any format/rate/layout) to s16le mono @ 16 kHz.

    av frames from webrtc are typically s16 stereo @ 48 kHz.
    We downsample to 16 kHz mono using numpy (simple decimation).
    """
    import av as _av

    # Reformat to s16 if needed
    if frame.format.name != "s16":
        frame = frame.reformat(format="s16")

    arr = frame.to_ndarray()  # shape: (channels, samples), dtype int16

    # Mix to mono
    if arr.shape[0] > 1:
        arr = arr.mean(axis=0, keepdims=True).astype(np.int16)
    else:
        arr = arr.astype(np.int16)

    # Resample to 16 kHz via simple decimation (webrtc delivers 48 kHz typically)
    src_rate = frame.sample_rate or 48000
    if src_rate != SAMPLE_RATE and src_rate > 0:
        ratio  = src_rate / SAMPLE_RATE
        n_out  = max(1, int(arr.shape[1] / ratio))
        indices = (np.arange(n_out) * ratio).astype(int)
        indices = np.clip(indices, 0, arr.shape[1] - 1)
        arr = arr[:, indices]

    return arr.flatten().astype(np.int16).tobytes()
