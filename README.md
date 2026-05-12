# Sarvam AI — STT & TTS Tester

A modular Streamlit app for testing every Sarvam AI speech API in one place:  
live mic transcription, batch jobs, REST synthesis, HTTP streaming, and real-time WebSocket TTS.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Coverage](#api-coverage)
  - [Speech-to-Text (STT)](#speech-to-text-stt)
  - [Text-to-Speech (TTS)](#text-to-speech-tts)
- [The Core Problem: Real-time Streaming in Streamlit](#the-core-problem-real-time-streaming-in-streamlit)
  - [Why it's hard](#why-its-hard)
  - [STT Live Streaming — Solution Architecture](#stt-live-streaming--solution-architecture)
  - [TTS Live Streaming — Solution Architecture](#tts-live-streaming--solution-architecture)
  - [Design decisions](#design-decisions)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Requirements](#requirements)

---

## Features

| Category | What's covered |
|---|---|
| STT REST | Single-file sync transcription (≤30 s), all 5 modes |
| STT Batch | Job-based long-audio (up to 1 h), diarization, timestamps |
| STT Live | Real-time browser mic → WebSocket → live transcript, all 5 modes |
| TTS REST | Basic, voice selection, full advanced params (model-specific) |
| TTS HTTP Stream | Chunked binary stream, progressive byte delivery |
| TTS WS One-shot | Full text → WebSocket → collect all audio → play |
| TTS WS Live | Persistent WebSocket, sentence-by-sentence, Web Audio API gapless playback |

---

## Project Structure

```
sarvam_app/
├── app.py              # Streamlit UI — two-column layout (STT | TTS)
├── stt.py              # All STT logic: REST, Batch, StreamingSession
├── tts.py              # All TTS logic: REST, HTTP stream, TtsStreamingSession
├── requirements.txt
├── .env.example
└── README.md
```

All secrets are loaded from `.env`. No keys in code.

---

## Quick Start

```bash
# 1. Clone / copy files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env:
#   SARVAM_API_KEY=your_key_here

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501` with STT on the left and TTS on the right.

> **Microphone access** — the Live Streaming STT tab uses WebRTC (`streamlit-webrtc`).  
> Your browser will prompt for mic permission when you click **Start recording**.  
> This requires HTTPS in production (localhost works without it).

---

## API Coverage

### Speech-to-Text (STT)

Model: **`saaras:v3`** (recommended, used everywhere)

#### Modes

| Mode | Output |
|---|---|
| `transcribe` | Original language text |
| `translate` | English translation |
| `verbatim` | Word-for-word including fillers |
| `translit` | Romanised (Latin script) |
| `codemix` | Code-mixed output (e.g. Hinglish) |

#### REST API tab

Upload any audio file (WAV, MP3, OGG, FLAC, AAC, M4A, WEBM, ≤30 s).  
Calls `client.speech_to_text.transcribe(file, model, mode, language_code)` synchronously.  
Returns `transcript`, detected `language_code`, and `request_id`.

#### Batch API tab

Upload audio up to 1 hour. Supports speaker diarization (up to 8 speakers).

Flow:
```
create_job(model, mode, language_code, with_diarization, num_speakers)
  → upload_files([tmp_path])          # SDK PUTs to S3 signed URL
  → start()
  → wait_until_complete(poll=5s)      # blocks until COMPLETED or FAILED
  → download_outputs(output_dir)      # fetches result JSON
  → get_file_results()                # per-file success/failure map
```

The uploaded file is written to a named temp file (SDK requires real paths), then deleted after upload.

#### Live Streaming tab

Real-time mic → WebSocket transcription.  
See [STT Live Streaming — Solution Architecture](#stt-live-streaming--solution-architecture) below.

---

### Text-to-Speech (TTS)

Default model: **`bulbul:v3`**

#### Advanced parameters by model

| Parameter | `bulbul:v2` | `bulbul:v3` |
|---|---|---|
| `pace` | ✅ 0.3–3.0 | ✅ 0.5–2.0 |
| `pitch` | ✅ | ❌ |
| `loudness` | ✅ | ❌ |
| `temperature` | ❌ | ✅ 0.01–1.0 |
| `speech_sample_rate` | ❌ | ✅ up to 48 kHz |
| `enable_preprocessing` | ❌ | ✅ |

#### REST tabs (Basic / Voice / Advanced)

`client.text_to_speech.convert(...)` returns base64-encoded audio in `response.audios[0]`.  
The app decodes and renders an HTML5 `<audio>` player + download button.

#### HTTP Stream tab

`client.text_to_speech.convert_stream(...)` returns `Iterator[bytes]` — raw binary, no base64 decode needed. Chunks are collected into a buffer while a live byte counter updates. Max text: 3500 chars.

#### WS — One-shot tab

Opens a `TtsStreamingSession`, sends the full text as a single `convert()` call, collects all `AudioOutput` chunks, then plays the assembled audio. Same parameters as REST Advanced.

#### WS — Live Streaming tab

Persistent WebSocket connection. Type and send sentences one at a time (or batch).  
Each sentence is synthesized and played immediately as chunks arrive via Web Audio API.  
See [TTS Live Streaming — Solution Architecture](#tts-live-streaming--solution-architecture) below.

---

## The Core Problem: Real-time Streaming in Streamlit

### Why it's hard

Streamlit's execution model is fundamentally **synchronous and stateless**: every user interaction rerenders the entire script from top to bottom in a single thread. This creates three concrete obstacles for real-time audio streaming:

1. **No persistent connections.** A Sarvam STT/TTS WebSocket must stay open for seconds or minutes across many rerender cycles. Streamlit has no concept of a long-lived object that survives reruns — only `st.session_state` does.

2. **No async execution.** The Sarvam SDK's streaming APIs require `AsyncSarvamAI` with `async with` context managers and `async for` loops. You cannot call `asyncio.run()` from inside Streamlit's synchronous execution context without blocking the UI, and you cannot call it from inside `aiortc`'s event loop (which already owns a running loop).

3. **No direct browser → Python audio bridge.** To capture live mic audio you need `getUserMedia` in the browser, but Streamlit has no built-in component for this. Raw PCM from the browser must cross the WebRTC boundary into Python, be resampled, and fed to the Sarvam API — all while Streamlit is rerending on a 500ms poll cycle.

4. **No progressive audio playback.** `st.audio()` requires the full audio file upfront. Playing TTS chunks as they arrive from a WebSocket requires Web Audio API in the browser, which Streamlit also has no native support for.

---

### STT Live Streaming — Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  BROWSER                                                            │
│                                                                     │
│  getUserMedia (mic)                                                 │
│       │                                                             │
│       ▼                                                             │
│  WebRTC peer connection  ◄──── streamlit-webrtc (aiortc)           │
└────────────────┬────────────────────────────────────────────────────┘
                 │  av.AudioFrame (s16, 48 kHz stereo)
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PYTHON — aiortc event loop                                         │
│                                                                     │
│  AudioProcessorBase.recv_queued(frames)                             │
│       │                                                             │
│       ▼                                                             │
│  _frame_to_pcm16_mono()                                             │
│    • reformat to s16 if needed                                      │
│    • mix channels to mono                                           │
│    • decimate from 48 kHz → 16 kHz (numpy integer indexing)        │
│       │                                                             │
│       ▼  put_nowait (non-blocking, drops if full)                   │
│  audio_queue: Queue[bytes]  (maxsize=200, ~20 s buffer)             │
└────────────────┬────────────────────────────────────────────────────┘
                 │  bytes: s16le PCM @ 16 kHz mono
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PYTHON — StreamingSession (daemon thread, own asyncio loop)        │
│                                                                     │
│  _send_loop (async task)                                            │
│    • drains audio_queue via run_in_executor (non-blocking bridge)   │
│    • accumulates into 100 ms / 3200-byte chunks                     │
│    • base64-encodes each chunk                                      │
│    • await ws.transcribe(b64, encoding="audio/x-raw", sr=16000)     │
│    • on flush sentinel → await ws.flush()  (forces VAD trigger)     │
│                                                                     │
│  _recv_loop (async task, concurrent with _send_loop)                │
│    • async for message in ws                                        │
│    • extracts transcript string from dict or typed response         │
│    • result_queue.put({"transcript": text})                         │
│    • also handles {"signal": "speech_start"/"speech_end"}           │
│                                                                     │
│  AsyncSarvamAI STT WebSocket (persistent for full session duration) │
│    model="saaras:v3", mode=<selected>, flush_signal="true"          │
│    high_vad_sensitivity="true", input_audio_codec="pcm_s16le"       │
└────────────────┬────────────────────────────────────────────────────┘
                 │  {"transcript": str}
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STREAMLIT — main thread                                            │
│                                                                     │
│  st_autorefresh(interval=500ms)  ← only active while recording     │
│       │                                                             │
│       ▼  on each rerender                                           │
│  session.drain_results()  →  append to st.session_state[segments]  │
│       │                                                             │
│       ▼                                                             │
│  st.text_area(value=" ".join(segments))  — live update             │
└─────────────────────────────────────────────────────────────────────┘
```

**Key mechanics:**

- `StreamingSession` is created once and stored in `st.session_state`. It survives reruns because session state persists across the Streamlit script lifecycle.
- `recv_queued` runs inside aiortc's own event loop. It must never block and must never call `asyncio.run()`. It only does `queue.put_nowait()`.
- The background daemon thread creates `asyncio.new_event_loop()` — completely isolated from both Streamlit's thread and aiortc's thread. This is the only safe way to run `async with` across multiple Streamlit reruns.
- `run_in_executor(None, lambda: queue.get(timeout=0.1))` bridges the sync queue into the async send loop without blocking the event loop.
- `flush_signal="true"` + explicit `ws.flush()` forces the Sarvam VAD to emit a transcript immediately instead of waiting for natural silence, giving sub-second latency on deliberate pauses.

---

### TTS Live Streaming — Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  STREAMLIT — main thread                                            │
│                                                                     │
│  User types sentence → st.text_input → "Send sentence" button      │
│       │                                                             │
│       ▼                                                             │
│  session.send_text(sentence)  →  text_queue.put(sentence)           │
│                                                                     │
│  st_autorefresh(interval=400ms)  ← only while WS is open           │
│       │                                                             │
│       ▼  on each rerender                                           │
│  session.drain_audio()  →  base64-encode each chunk                 │
│       │                                                             │
│       ▼  accumulate in st.session_state[chunks_b64]                 │
│  _web_audio_player(chunks_b64, codec)                               │
└────────────────┬────────────────────────────────────────────────────┘
                 │  str (sentence text)
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PYTHON — TtsStreamingSession (daemon thread, own asyncio loop)     │
│                                                                     │
│  _send_loop (async task)                                            │
│    • drains text_queue via run_in_executor                          │
│    • None  →  poison pill, stop                                     │
│    • ""    →  flush-only sentinel (explicit flush without text)     │
│    • text  →  await ws.convert(text)                                │
│               await ws.flush()   ← flush per sentence for low TTFB │
│                                                                     │
│  _recv_loop (async task, concurrent with _send_loop)                │
│    • async for message in ws                                        │
│    • isinstance(message, AudioOutput)                               │
│        → base64.b64decode(message.data.audio)                       │
│        → audio_queue.put(chunk_bytes)                               │
│    • isinstance(message, EventResponse)                             │
│        → if event_type == "final" and not running → break           │
│                                                                     │
│  AsyncSarvamAI TTS WebSocket                                        │
│    connect(model, send_completion_event="true")                     │
│    configure(language, speaker, pace, pitch, loudness, ...)  once   │
│    — connection stays open until session.stop()                     │
└────────────────┬────────────────────────────────────────────────────┘
                 │  bytes (decoded MP3/WAV chunk)
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BROWSER — Web Audio API  (injected via st.components.v1.html)      │
│                                                                     │
│  AudioContext                                                       │
│  nextStart = ctx.currentTime + 0.05  (initial scheduling offset)   │
│                                                                     │
│  for each base64 chunk:                                             │
│    atob(b64)  →  Uint8Array                                         │
│    ctx.decodeAudioData(buffer)  →  AudioBuffer                      │
│    src = ctx.createBufferSource()                                   │
│    src.buffer = audioBuf                                            │
│    src.start(max(nextStart, ctx.currentTime))  ← schedule ahead    │
│    nextStart += audioBuf.duration               ← chain seamlessly  │
│                                                                     │
│  Result: gapless, low-latency playback of each chunk               │
│  as soon as it is decoded — no waiting for full audio              │
└─────────────────────────────────────────────────────────────────────┘
```

**Key mechanics:**

- `TtsStreamingSession` is stored in `st.session_state` keyed by panel. It persists across reruns and multiple `send_text()` calls — the same WebSocket connection is reused for every sentence, exactly like a production voice agent.
- `ws.configure()` is sent **once** after `connect()`. All subsequent `convert()` calls reuse that configuration. Reconnecting per sentence would add ~200–500 ms handshake overhead per utterance.
- `await ws.flush()` is called **after every sentence** (not just at the end). This tells Sarvam's TTS to process the current buffer immediately, delivering the first audio chunk within ~150–300 ms of sending the text.
- `send_completion_event="true"` causes the server to emit an `EventResponse(event_type="final")` when synthesis is complete, giving a clean, reliable termination signal instead of timeout-based polling.
- The Web Audio API `AudioContext` schedules each decoded chunk at `nextStart`, which is advanced by the chunk's duration after each scheduling. This produces gapless playback even when chunks arrive with irregular timing.
- `st.components.v1.html()` re-injects the player HTML on every rerender. Chunks already played are re-scheduled, but because `nextStart` is recalculated from `ctx.currentTime` on each injection, only new chunks add actual sound. Previously played chunks are silently skipped by the browser's audio scheduler.

---

### Design decisions

| Decision | Why |
|---|---|
| Dedicated `threading.Thread` + `asyncio.new_event_loop()` per session | Cannot call `asyncio.run()` inside aiortc's loop or Streamlit's sync thread. A fully isolated loop is the only safe option. |
| `queue.Queue` as the async↔sync bridge | Thread-safe, works across all three execution contexts (aiortc loop, background thread, Streamlit main thread) without any shared locks. |
| `run_in_executor(None, queue.get(timeout=…))` | Bridges a blocking `queue.get()` into an async context without blocking the event loop itself. |
| `put_nowait` with `maxsize=200` on STT audio queue | Applies backpressure — if the Sarvam WS falls behind, new audio is dropped rather than queuing indefinitely and causing memory growth. |
| `st_autorefresh` only while streaming is active | Prevents unnecessary reruns when the user is on a non-streaming tab or has stopped recording. |
| NumPy integer-index decimation for PCM resampling | Avoids `scipy` as a dependency. Simple decimation is sufficient for speech intelligibility at 16 kHz. |
| Web Audio API over `st.audio()` for TTS chunks | `st.audio()` requires the complete file. `AudioContext.decodeAudioData` + scheduled `BufferSource` nodes is the only browser-native way to achieve gapless progressive playback. |
| `ws.flush()` per sentence in TTS | Minimises time-to-first-byte per utterance. Without per-sentence flush, Sarvam buffers until `max_chunk_length` is reached, adding latency. |

---

## Configuration

### `.env` file

```env
SARVAM_API_KEY=your_sarvam_api_key_here
```

The key can also be entered directly in the sidebar — it overrides the `.env` value.

### Streamlit config (optional)

For production or HTTPS deployments, add a `.streamlit/config.toml`:

```toml
[server]
address = "0.0.0.0"
port = 8501

[browser]
gatherUsageStats = false
```

> **WebRTC / microphone note:** `streamlit-webrtc` requires a STUN server for the ICE negotiation.  
> The app uses `stun:stun.l.google.com:19302` (public, no setup needed).  
> For production deployments behind NAT, you may need a TURN server.

---

## Error Handling

All API calls catch `ApiError` from `sarvamai.core.api_error` and map status codes to human-readable messages:

| HTTP | Meaning | Action |
|---|---|---|
| 400 | Bad request | Check audio format / text params |
| 403 | Invalid API key | Verify key in dashboard |
| 422 | Unprocessable entity | Unsupported audio format, text too long, or invalid params |
| 429 | Rate limit / quota | Wait and retry; consider upgrading plan |
| 500 | Sarvam server error | Retry; contact support if persistent |
| 503 | Service overloaded | Retry with exponential backoff |

Streaming sessions surface errors through their result/audio queues so the UI can display them without crashing the background thread. WebSocket disconnects are caught as generic `Exception` and reported via the same path.

---

## Requirements

```
sarvamai                # Sarvam AI Python SDK
streamlit               # UI framework
python-dotenv           # .env loading
requests                # Batch job S3 upload
streamlit-webrtc        # Browser mic capture via WebRTC
streamlit-autorefresh   # Polling rerun trigger
numpy                   # PCM resampling
av                      # Audio frame decoding (PyAV, pulled by streamlit-webrtc)
aiortc                  # WebRTC peer connection (pulled by streamlit-webrtc)
```

Install:

```bash
pip install -r requirements.txt
```