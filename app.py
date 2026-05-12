import base64
import os
import queue
import time

import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

import stt as stt_module
import tts as tts_module

load_dotenv()
st.set_page_config(page_title="Sarvam AI — STT & TTS Tester", layout="wide", page_icon="🎙")

_ENV_KEY = os.getenv("SARVAM_API_KEY", "")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _key() -> str:
    return st.session_state.get("api_key_input") or _ENV_KEY

def _require_key() -> str:
    k = _key()
    if not k:
        st.error("⚠️ No API key. Set SARVAM_API_KEY in .env or enter it in the sidebar.")
        st.stop()
    return k

def _require_client():
    return stt_module.make_client(_require_key())

def _audio_player(audio_bytes: bytes, codec: str, key: str = "player"):
    mime = {"mp3":"audio/mpeg","wav":"audio/wav","flac":"audio/flac",
            "aac":"audio/aac","opus":"audio/ogg; codecs=opus"}.get(codec,"audio/wav")
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio controls style="width:100%">'
        f'<source src="data:{mime};base64,{b64}" type="{mime}"></audio>',
        unsafe_allow_html=True,
    )
    st.download_button("⬇ Download", audio_bytes,
                       file_name=f"output.{codec}", mime=mime,
                       use_container_width=True, key=key)

def _show_raw(data: dict):
    with st.expander("🔍 Raw response"):
        st.json({k: v for k, v in data.items() if k != "audio_bytes"})


# ── Web Audio Player component ────────────────────────────────────────────────

def _web_audio_player(chunks_b64: list[str], codec: str):
    """
    Inject a Web Audio API player that decodes and plays MP3/WAV chunks
    using AudioContext for gapless progressive playback.
    Each call appends new chunks to an in-page queue and plays immediately.
    """
    chunks_json = "[" + ",".join(f'"{c}"' for c in chunks_b64) + "]"
    mime = "audio/mpeg" if codec == "mp3" else f"audio/{codec}"
    html = f"""
<div id="waudio-status" style="font-size:0.8em;color:#888;margin-bottom:4px;">
  ⏳ Buffering audio…
</div>
<div id="waudio-bar" style="background:#e0e0e0;border-radius:4px;height:6px;width:100%;margin-bottom:8px;">
  <div id="waudio-progress" style="background:#4CAF50;height:6px;width:0%;border-radius:4px;transition:width 0.2s;"></div>
</div>
<script>
(function() {{
  const chunks  = {chunks_json};
  const mime    = "{mime}";
  const status  = document.getElementById("waudio-status");
  const progBar = document.getElementById("waudio-progress");

  if (!chunks.length) {{ status.textContent = "No audio data."; return; }}

  const ctx       = new (window.AudioContext || window.webkitAudioContext)();
  let   nextStart = ctx.currentTime + 0.05;
  let   played    = 0;

  async function decodeAndSchedule(b64) {{
    const binary = atob(b64);
    const buf    = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) buf[i] = binary.charCodeAt(i);
    try {{
      const audioBuf = await ctx.decodeAudioData(buf.buffer.slice(0));
      const src      = ctx.createBufferSource();
      src.buffer     = audioBuf;
      src.connect(ctx.destination);
      src.start(Math.max(nextStart, ctx.currentTime));
      nextStart = Math.max(nextStart, ctx.currentTime) + audioBuf.duration;
      played++;
      progBar.style.width = (played / chunks.length * 100).toFixed(1) + "%";
      if (played === chunks.length) status.textContent = "✅ Playback complete";
    }} catch(e) {{
      console.warn("Decode error:", e);
    }}
  }}

  status.textContent = `▶ Playing ${{chunks.length}} chunk(s)…`;
  chunks.forEach(decodeAndSchedule);
}})();
</script>
"""
    st.components.v1.html(html, height=60)


# ── Live TTS WebSocket component ──────────────────────────────────────────────

def _progressive_tts_player(session_key: str):
    """
    Poll the TtsStreamingSession stored at session_key, drain audio chunks,
    accumulate in session_state, and re-render the Web Audio player.
    """
    session: tts_module.TtsStreamingSession = st.session_state.get(session_key)
    if session is None:
        return

    acc_key = f"{session_key}_chunks"
    if acc_key not in st.session_state:
        st.session_state[acc_key] = []

    new_chunks = session.drain_audio()
    if new_chunks:
        for c in new_chunks:
            st.session_state[acc_key].append(base64.b64encode(c).decode())

    chunks_b64: list[str] = st.session_state[acc_key]

    if session.error:
        st.error(f"❌ {session.error}")
    elif chunks_b64:
        codec = session.output_audio_codec
        st.caption(f"{len(chunks_b64)} chunk(s) received · codec: {codec}")
        _web_audio_player(chunks_b64, codec)

        # Full audio download once session ends
        if not (session._thread and session._thread.is_alive()):
            all_audio = b"".join(
                base64.b64decode(c) for c in chunks_b64
            )
            st.download_button("⬇ Download full audio", all_audio,
                               file_name=f"tts_ws.{codec}",
                               mime="audio/mpeg" if codec == "mp3" else f"audio/{codec}",
                               use_container_width=True,
                               key=f"{session_key}_dl")
    else:
        st.caption("Waiting for audio…")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔑 API Key")
    st.text_input("Sarvam API Key", value=_ENV_KEY, type="password",
                  key="api_key_input",
                  help="Overrides SARVAM_API_KEY from .env")
    st.caption("STT model: `saaras:v3`  ·  TTS default: `bulbul:v3`")
    st.divider()
    st.caption("[Sarvam Docs](https://docs.sarvam.ai/api-reference-docs/getting-started/welcome)")


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🎙 Sarvam AI — STT & TTS Tester")
col_stt, col_tts = st.columns(2, gap="large")


# ═════════════════════════════════════════════════════════════════════════════
# LEFT — Speech-to-Text
# ═════════════════════════════════════════════════════════════════════════════

with col_stt:
    st.header("🎤 Speech-to-Text")

    stt_mode = st.selectbox("Mode", stt_module.MODES, key="stt_mode")
    stt_path = st.radio("API path", ["REST API", "Batch API", "Live Streaming"],
                        horizontal=True, key="stt_path")
    stt_lang = st.selectbox("Language", stt_module.LANGUAGES, key="stt_lang")
    st.caption(f"Model: `{stt_module.STT_MODEL}`")
    st.divider()

    # ── REST ─────────────────────────────────────────────────────────────────
    if stt_path == "REST API":
        uploaded = st.file_uploader("Upload audio (≤30 s)",
                                    type=["wav","mp3","ogg","flac","aac","m4a","webm"],
                                    key="stt_upload_rest")
        audio_bytes_in = uploaded.read() if uploaded else None
        audio_fname    = uploaded.name   if uploaded else "audio.wav"
        if audio_bytes_in:
            st.audio(audio_bytes_in)

        if st.button("▶ Transcribe", key="run_stt_rest",
                     use_container_width=True, type="primary"):
            if not audio_bytes_in:
                st.warning("Upload an audio file first.")
            else:
                with st.spinner("Transcribing…"):
                    result = stt_module.stt_rest(
                        _require_client(), audio_bytes_in, audio_fname,
                        stt_mode, stt_lang)
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Done")
                    st.text_area("📝 Transcript", result.get("transcript",""),
                                 height=120, key="stt_rest_out")
                    if result.get("language_code"):
                        st.caption(f"Detected: `{result['language_code']}`")
                _show_raw(result)

    # ── Batch ─────────────────────────────────────────────────────────────────
    elif stt_path == "Batch API":
        c1, c2 = st.columns(2)
        with_diarize = c1.checkbox("Speaker diarization", key="stt_diarize")
        num_spk      = c2.number_input("# Speakers", 1, 8, 2, key="stt_speakers",
                                       disabled=not with_diarize)
        uploaded = st.file_uploader("Upload audio (up to 1 h)",
                                    type=["wav","mp3","ogg","flac","aac","m4a","webm"],
                                    key="stt_upload_batch")
        audio_bytes_in = uploaded.read() if uploaded else None
        audio_fname    = uploaded.name   if uploaded else "audio.wav"
        if audio_bytes_in:
            st.audio(audio_bytes_in)

        if st.button("▶ Submit Batch Job", key="run_stt_batch",
                     use_container_width=True, type="primary"):
            if not audio_bytes_in:
                st.warning("Upload an audio file first.")
            else:
                log_ph = st.empty()
                logs: list[str] = []
                def _log(m):
                    logs.append(m); log_ph.info("\n".join(logs[-5:]))
                with st.spinner("Running batch job…"):
                    result = stt_module.stt_batch(
                        _require_client(), audio_bytes_in, audio_fname,
                        stt_mode, stt_lang,
                        with_diarization=with_diarize,
                        num_speakers=int(num_spk) if with_diarize else None,
                        progress_cb=_log)
                log_ph.empty()
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Done")
                    st.text_area("📝 Transcript", result.get("transcript",""),
                                 height=120, key="stt_batch_out")
                    if result.get("file_results"):
                        with st.expander("File results"):
                            st.json(result["file_results"])
                _show_raw(result)

    # ── Live Streaming ─────────────────────────────────────────────────────
    else:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        st.caption(
            "🎙 Click **START** to open your microphone. "
            "Transcript updates every ~500 ms while recording."
        )

        # Session-state keys for this panel
        _SS_KEY    = "stt_stream_session"   # StreamingSession object
        _TRANS_KEY = "stt_live_transcript"  # accumulated transcript string
        _SEGS_KEY  = "stt_live_segments"    # list of segment strings

        # Initialise session state
        if _TRANS_KEY not in st.session_state:
            st.session_state[_TRANS_KEY] = ""
        if _SEGS_KEY not in st.session_state:
            st.session_state[_SEGS_KEY] = []

        # ── Start / Stop controls ───────────────────────────────────────
        c1, c2 = st.columns(2)
        start_clicked = c1.button("🔴 Start recording", key="stt_start",
                                   use_container_width=True)
        stop_clicked  = c2.button("⏹ Stop & flush",    key="stt_stop",
                                   use_container_width=True)

        # ── Manage StreamingSession lifecycle ───────────────────────────
        if start_clicked:
            # Tear down any previous session
            old: stt_module.StreamingSession = st.session_state.get(_SS_KEY)
            if old: old.stop()
            # Reset transcript
            st.session_state[_TRANS_KEY] = ""
            st.session_state[_SEGS_KEY]  = []
            # Create and start new session
            session = stt_module.StreamingSession(
                api_key        = _require_key(),
                language_code  = stt_lang,
                mode           = stt_mode,
            )
            session.start()
            st.session_state[_SS_KEY] = session

        if stop_clicked:
            session: stt_module.StreamingSession = st.session_state.get(_SS_KEY)
            if session:
                session.flush()   # trigger final transcript
                time.sleep(0.8)   # give WS a moment to respond
                session.stop()
                st.session_state[_SS_KEY] = None

        # ── webrtc_streamer — mic capture ───────────────────────────────
        session: stt_module.StreamingSession = st.session_state.get(_SS_KEY)

        def _audio_processor_factory():
            if session is None:
                # Return a passthrough processor when not recording
                from streamlit_webrtc import AudioProcessorBase
                import av as _av
                class _Passthrough(AudioProcessorBase):
                    def recv(self, frame): return frame
                return _Passthrough
            return stt_module.make_audio_processor(session)

        webrtc_ctx = webrtc_streamer(
            key           = "stt_webrtc",
            mode          = WebRtcMode.SENDONLY,
            audio_processor_factory = _audio_processor_factory,
            media_stream_constraints = {"audio": True, "video": False},
            sendback_audio = False,
            rtc_configuration = {
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        # ── Auto-refresh polls transcript_queue while recording ─────────
        is_recording = (session is not None and session._running
                        and webrtc_ctx and webrtc_ctx.state.playing)

        if is_recording:
            st_autorefresh(interval=500, key="stt_autorefresh")

        # ── Drain results from the session ──────────────────────────────
        if session:
            for item in session.drain_results():
                if "transcript" in item and item["transcript"]:
                    st.session_state[_SEGS_KEY].append(item["transcript"])
                    st.session_state[_TRANS_KEY] = " ".join(
                        st.session_state[_SEGS_KEY])
                elif "error" in item:
                    st.error(f"❌ {item['error']}")

        # ── Display live transcript ─────────────────────────────────────
        st.divider()
        status_text = "🔴 Recording…" if is_recording else (
            "⏹ Stopped" if session is None else "⚙️ Session ready — click START")
        st.caption(status_text)

        transcript_ph = st.empty()
        transcript_ph.text_area(
            "📝 Live transcript",
            value  = st.session_state[_TRANS_KEY],
            height = 160,
            key    = "stt_live_out",
        )

        if st.session_state[_SEGS_KEY]:
            with st.expander(f"Segments ({len(st.session_state[_SEGS_KEY])})"):
                for i, seg in enumerate(st.session_state[_SEGS_KEY], 1):
                    st.write(f"{i}. {seg}")

        if st.button("🗑 Clear transcript", key="stt_clear", use_container_width=True):
            st.session_state[_TRANS_KEY] = ""
            st.session_state[_SEGS_KEY]  = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT — Text-to-Speech
# ═════════════════════════════════════════════════════════════════════════════

with col_tts:
    st.header("🔊 Text-to-Speech")

    tts_control = st.radio(
        "Control",
        ["REST — Basic", "REST — Voice", "REST — Advanced",
         "HTTP Stream", "WS — One-shot", "WS — Live Streaming"],
        horizontal=False, key="tts_control",
    )
    st.divider()

    # ── shared inputs ─────────────────────────────────────────────────────
    tts_lang = st.selectbox("Language", tts_module.TTS_LANGUAGES, key="tts_lang")

    # defaults (overridden below)
    tts_speaker   = "shubh"
    tts_model     = "bulbul:v3"
    tts_codec     = "wav"
    tts_bitrate   = "128k"
    tts_pace      = None
    tts_pitch     = None
    tts_loudness  = None
    tts_temp      = None
    tts_sr        = None
    tts_preproc   = None

    # ── REST Basic ────────────────────────────────────────────────────────
    if tts_control == "REST — Basic":
        tts_text  = st.text_area("Text", "Welcome to Sarvam AI!", height=80, key="tts_text_basic")
        tts_model = st.selectbox("Model", tts_module.TTS_MODELS, key="tb_model")
        tts_codec = st.selectbox("Codec", tts_module.REST_CODECS, key="tb_codec")

        if st.button("▶ Synthesize", key="run_tts_basic",
                     use_container_width=True, type="primary"):
            if not tts_text.strip(): st.warning("Enter text first.")
            else:
                with st.spinner("Synthesizing…"):
                    r = tts_module.tts_rest(_require_client(), tts_text, tts_lang,
                                            model=tts_model, output_audio_codec=tts_codec)
                if "error" in r: st.error(f"❌ {r['error']}")
                else:
                    st.success("✅ Done")
                    _audio_player(r["audio_bytes"], tts_codec, "pl_basic")
                _show_raw(r)

    # ── REST Voice ────────────────────────────────────────────────────────
    elif tts_control == "REST — Voice":
        tts_text    = st.text_area("Text", "नमस्ते! मैं सर्वम एआई हूँ।", height=80, key="tts_text_voice")
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"), key="tv_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="tv_model")
        tts_codec   = st.selectbox("Codec", tts_module.REST_CODECS, key="tv_codec")

        if st.button("▶ Synthesize", key="run_tts_voice",
                     use_container_width=True, type="primary"):
            if not tts_text.strip(): st.warning("Enter text first.")
            else:
                with st.spinner("Synthesizing…"):
                    r = tts_module.tts_rest(_require_client(), tts_text, tts_lang,
                                            speaker=tts_speaker, model=tts_model,
                                            output_audio_codec=tts_codec)
                if "error" in r: st.error(f"❌ {r['error']}")
                else:
                    st.success("✅ Done")
                    _audio_player(r["audio_bytes"], tts_codec, "pl_voice")
                _show_raw(r)

    # ── REST Advanced ─────────────────────────────────────────────────────
    elif tts_control == "REST — Advanced":
        tts_text    = st.text_area("Text", "भारत की संस्कृति विश्व की सबसे प्राचीन है।",
                                   height=80, key="tts_text_adv")
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"), key="ta_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="ta_model")
        tts_codec   = st.selectbox("Codec", tts_module.REST_CODECS, key="ta_codec")
        tts_pace    = st.slider("Pace", 0.5, 2.0, 1.0, 0.05, key="ta_pace")
        if tts_model == "bulbul:v2":
            c1, c2 = st.columns(2)
            tts_pitch    = c1.slider("Pitch",    -0.75, 0.75, 0.0, 0.05, key="ta_pitch")
            tts_loudness = c2.slider("Loudness",  0.5,  3.0,  1.0, 0.05, key="ta_loudness")
        else:
            c1, c2 = st.columns(2)
            tts_temp    = c1.slider("Temperature", 0.01, 1.0, 0.6, 0.01, key="ta_temp")
            tts_sr      = c2.select_slider("Sample rate",
                                           [8000,16000,22050,24000,32000,44100,48000],
                                           value=22050, key="ta_sr")
            tts_preproc = st.checkbox("Enable preprocessing", False, key="ta_preproc")

        if st.button("▶ Synthesize", key="run_tts_adv",
                     use_container_width=True, type="primary"):
            if not tts_text.strip(): st.warning("Enter text first.")
            else:
                with st.spinner("Synthesizing…"):
                    r = tts_module.tts_rest(_require_client(), tts_text, tts_lang,
                                            speaker=tts_speaker, model=tts_model,
                                            output_audio_codec=tts_codec,
                                            pitch=tts_pitch, loudness=tts_loudness,
                                            pace=tts_pace, temperature=tts_temp,
                                            speech_sample_rate=tts_sr,
                                            enable_preprocessing=tts_preproc)
                if "error" in r: st.error(f"❌ {r['error']}")
                else:
                    st.success("✅ Done")
                    _audio_player(r["audio_bytes"], tts_codec, "pl_adv")
                _show_raw(r)

    # ── HTTP Stream ───────────────────────────────────────────────────────
    elif tts_control == "HTTP Stream":
        tts_text    = st.text_area("Text (up to 3500 chars)",
                                   "नमस्ते! Sarvam AI में आपका स्वागत है।",
                                   height=80, key="tts_text_hs")
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"), key="hs_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="hs_model")
        tts_codec   = st.selectbox("Codec", tts_module.STREAM_CODECS, key="hs_codec")
        tts_bitrate = st.selectbox("Bitrate", tts_module.BITRATES, index=3, key="hs_bitrate")
        tts_pace    = st.slider("Pace", 0.5, 2.0, 1.0, 0.05, key="hs_pace")
        if tts_model == "bulbul:v3":
            c1, c2 = st.columns(2)
            tts_temp    = c1.slider("Temperature", 0.01, 1.0, 0.6, 0.01, key="hs_temp")
            tts_preproc = c2.checkbox("Preprocessing", False, key="hs_preproc")

        prog_ph = st.empty()
        if st.button("▶ Stream", key="run_tts_hs",
                     use_container_width=True, type="primary"):
            if not tts_text.strip(): st.warning("Enter text first.")
            else:
                def _prog(n): prog_ph.caption(f"Received {n:,} bytes…")
                with st.spinner("Streaming…"):
                    r = tts_module.tts_http_stream(
                        _require_client(), tts_text, tts_lang,
                        speaker=tts_speaker, model=tts_model,
                        output_audio_codec=tts_codec,
                        output_audio_bitrate=tts_bitrate,
                        pace=tts_pace, temperature=tts_temp,
                        speech_sample_rate=tts_sr,
                        enable_preprocessing=tts_preproc,
                        progress_cb=_prog)
                prog_ph.empty()
                if "error" in r: st.error(f"❌ {r['error']}")
                else:
                    st.success(f"✅ {r['chunks_received']} chunks received")
                    _audio_player(r["audio_bytes"], tts_codec, "pl_hs")
                _show_raw(r)

    # ── WS One-shot ───────────────────────────────────────────────────────
    elif tts_control == "WS — One-shot":
        st.caption("Opens a WebSocket, sends full text, collects all audio, plays on completion.")
        tts_text    = st.text_area("Text", "भारत की संस्कृति विश्व की सबसे प्राचीन "
                                   "और समृद्ध संस्कृतियों में से एक है।",
                                   height=80, key="tts_text_ws1")
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"), key="ws1_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="ws1_model")
        tts_codec   = st.selectbox("Codec", tts_module.WS_CODECS, key="ws1_codec")
        tts_bitrate = st.selectbox("Bitrate", tts_module.BITRATES, index=3, key="ws1_bitrate")
        c1, c2, c3 = st.columns(3)
        tts_pace    = c1.slider("Pace",    0.5, 2.0,  1.0, 0.05, key="ws1_pace")
        tts_pitch   = c2.slider("Pitch", -0.75, 0.75, 0.0, 0.05, key="ws1_pitch")
        tts_loudness= c3.slider("Loudness", 0.5, 3.0, 1.0, 0.05, key="ws1_loudness")

        log_ph = st.empty()
        if st.button("▶ Run", key="run_tts_ws1",
                     use_container_width=True, type="primary"):
            if not tts_text.strip(): st.warning("Enter text first.")
            else:
                logs: list[str] = []
                def _log(m): logs.append(m); log_ph.info("\n".join(logs[-4:]))
                with st.spinner("Synthesizing via WebSocket…"):
                    r = tts_module.tts_websocket_sync(
                        api_key=_key(), text=tts_text,
                        target_language_code=tts_lang,
                        speaker=tts_speaker, model=tts_model,
                        output_audio_codec=tts_codec,
                        output_audio_bitrate=tts_bitrate,
                        pace=float(tts_pace), pitch=float(tts_pitch),
                        loudness=float(tts_loudness),
                        progress_cb=_log)
                log_ph.empty()
                if "error" in r: st.error(f"❌ {r['error']}")
                else:
                    st.success(f"✅ {r['chunks_received']} chunks received")
                    _audio_player(r["audio_bytes"], tts_codec, "pl_ws1")
                _show_raw(r)

    # ── WS Live Streaming ─────────────────────────────────────────────────
    else:
        st.caption(
            "🎤 Send text sentence-by-sentence. Each sentence is synthesized and "
            "played immediately as chunks arrive — low-latency conversational flow."
        )

        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"), key="wsl_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="wsl_model")
        tts_codec   = st.selectbox("Codec", tts_module.WS_CODECS, key="wsl_codec")
        tts_bitrate = st.selectbox("Bitrate", tts_module.BITRATES, index=3, key="wsl_bitrate")
        c1, c2, c3 = st.columns(3)
        wsl_pace    = c1.slider("Pace",    0.5, 2.0,  1.0, 0.05, key="wsl_pace")
        wsl_pitch   = c2.slider("Pitch", -0.75, 0.75, 0.0, 0.05, key="wsl_pitch")
        wsl_loudness= c3.slider("Loudness", 0.5, 3.0, 1.0, 0.05, key="wsl_loudness")
        wsl_sr      = st.select_slider("Sample rate (Hz)",
                                       [8000,16000,22050,24000], value=22050, key="wsl_sr")
        wsl_preproc = st.checkbox("Enable preprocessing", False, key="wsl_preproc")

        st.divider()

        _WS_SESSION_KEY = "tts_ws_live_session"

        # ── Open / Close session controls ──────────────────────────────
        c1, c2 = st.columns(2)
        open_ws  = c1.button("🔌 Open WebSocket", key="tts_ws_open", use_container_width=True)
        close_ws = c2.button("❌ Close WebSocket", key="tts_ws_close", use_container_width=True)

        if open_ws:
            old: tts_module.TtsStreamingSession = st.session_state.get(_WS_SESSION_KEY)
            if old: old.stop()
            st.session_state[f"{_WS_SESSION_KEY}_chunks"] = []
            s = tts_module.TtsStreamingSession(
                api_key              = _require_key(),
                target_language_code = tts_lang,
                speaker              = tts_speaker,
                model                = tts_model,
                output_audio_codec   = tts_codec,
                output_audio_bitrate = tts_bitrate,
                pace                 = float(wsl_pace),
                pitch                = float(wsl_pitch),
                loudness             = float(wsl_loudness),
                speech_sample_rate   = int(wsl_sr),
                enable_preprocessing = bool(wsl_preproc),
            )
            s.start()
            st.session_state[_WS_SESSION_KEY] = s
            st.success("✅ WebSocket open")

        if close_ws:
            s: tts_module.TtsStreamingSession = st.session_state.get(_WS_SESSION_KEY)
            if s:
                s.stop()
                st.session_state[_WS_SESSION_KEY] = None
                st.info("WebSocket closed.")

        # ── Text input area ────────────────────────────────────────────
        ws_live_session: tts_module.TtsStreamingSession = st.session_state.get(_WS_SESSION_KEY)
        is_ws_open = ws_live_session is not None and ws_live_session._running

        st.caption("Status: " + ("🟢 WebSocket open — send sentences below"
                                  if is_ws_open else "⚫ WebSocket closed — click Open first"))

        sentence_input = st.text_input(
            "Type a sentence and press ↵ Send",
            placeholder="e.g. नमस्ते! आप कैसे हैं?",
            key="tts_ws_sentence",
            disabled=not is_ws_open,
        )
        send_sentence = st.button("↵ Send sentence", key="tts_ws_send",
                                   use_container_width=True,
                                   disabled=not is_ws_open)

        if send_sentence and sentence_input.strip() and is_ws_open:
            ws_live_session.send_text(sentence_input.strip())
            st.success(f"Sent: {chr(39)}{sentence_input.strip()}{chr(39)}")

        # ── Multi-sentence batch sender ────────────────────────────────
        with st.expander("📋 Send multiple sentences at once"):
            multi_text = st.text_area(
                "Sentences (one per line)",
                placeholder="Line 1\nLine 2\nLine 3",
                height=100, key="tts_ws_multi",
            )
            if st.button("↵ Send all", key="tts_ws_send_multi",
                          disabled=not is_ws_open):
                lines = [l.strip() for l in multi_text.splitlines() if l.strip()]
                for line in lines:
                    ws_live_session.send_text(line)
                st.success(f"Sent {len(lines)} sentence(s)")

        # ── Progressive audio player ───────────────────────────────────
        st.divider()
        st.caption("🔊 Audio (updates as chunks arrive)")

        # Auto-refresh while WS is open and producing audio
        if is_ws_open:
            st_autorefresh(interval=400, key="tts_ws_autorefresh")

        _progressive_tts_player(_WS_SESSION_KEY)

