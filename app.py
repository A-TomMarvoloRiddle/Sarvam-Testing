import base64
import os

import streamlit as st
from dotenv import load_dotenv

import stt as stt_module
import tts as tts_module

# ── Bootstrap ─────────────────────────────────────────────────────────────────

load_dotenv()
st.set_page_config(page_title="Sarvam AI — STT & TTS Tester", layout="wide", page_icon="🎙")

_ENV_KEY = os.getenv("SARVAM_API_KEY", "")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _active_key() -> str:
    return st.session_state.get("api_key_input") or _ENV_KEY


def _require_client():
    key = _active_key()
    if not key:
        st.error("⚠️ No API key. Set SARVAM_API_KEY in .env or enter it in the sidebar.")
        st.stop()
    return stt_module.make_client(key)


def _audio_player(audio_bytes: bytes, codec: str):
    """Inline HTML5 audio + download button."""
    mime_map = {
        "mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac",
        "opus": "audio/ogg; codecs=opus", "aac": "audio/aac",
        "ogg": "audio/ogg", "linear16": "audio/wav", "mulaw": "audio/basic",
    }
    mime = mime_map.get(codec, "audio/wav")
    b64  = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio controls style="width:100%">'
        f'<source src="data:{mime};base64,{b64}" type="{mime}"></audio>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "⬇ Download audio", audio_bytes,
        file_name=f"output.{codec}", mime=mime, use_container_width=True,
    )


def _show_raw(data: dict, exclude: str = "audio_bytes"):
    with st.expander("🔍 Raw response / metadata"):
        st.json({k: v for k, v in data.items() if k != exclude})


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔑 API Key")
    st.text_input(
        "Sarvam API Key", value=_ENV_KEY, type="password",
        key="api_key_input",
        help="Overrides SARVAM_API_KEY from .env",
    )
    st.caption("Model: **saaras:v3** (STT) · **bulbul:v3** (TTS, default)")
    st.divider()
    st.caption("Docs: [sarvam.ai/api-reference-docs](https://docs.sarvam.ai/api-reference-docs/getting-started/welcome)")


# ── Layout ─────────────────────────────────────────────────────────────────────

st.title("🎙 Sarvam AI — STT & TTS Tester")
col_stt, col_tts = st.columns(2, gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Speech-to-Text
# ══════════════════════════════════════════════════════════════════════════════

with col_stt:
    st.header("🎤 Speech-to-Text")

    # ── shared config ──────────────────────────────────────────────────────
    stt_mode = st.selectbox("Mode", stt_module.MODES, key="stt_mode")
    stt_path = st.radio("API path", ["REST API", "Batch API", "Streaming API"],
                        horizontal=True, key="stt_path")
    stt_lang = st.selectbox("Language", stt_module.LANGUAGES, key="stt_lang")
    st.caption(f"Model: `{stt_module.STT_MODEL}`")
    st.divider()

    # ── audio input ────────────────────────────────────────────────────────
    if stt_path == "REST API":
        uploaded = st.file_uploader(
            "Upload audio (≤30 s)", type=["wav","mp3","ogg","flac","aac","m4a","webm"],
            key="stt_upload_rest",
        )
        audio_bytes_in = uploaded.read() if uploaded else None
        audio_fname    = uploaded.name   if uploaded else "audio.wav"

    elif stt_path == "Batch API":
        st.caption("Supports up to 1-hour recordings.")
        c1, c2 = st.columns(2)
        with_diarization = c1.checkbox("Speaker diarization", key="stt_diarize")
        num_speakers = c2.number_input("# Speakers", 1, 8, 2, key="stt_speakers", disabled=not with_diarization)
        uploaded = st.file_uploader(
            "Upload audio (up to 1 h)", type=["wav","mp3","ogg","flac","aac","m4a","webm"],
            key="stt_upload_batch",
        )
        audio_bytes_in = uploaded.read() if uploaded else None
        audio_fname    = uploaded.name   if uploaded else "audio.wav"

    else:  # Streaming
        st.caption("⚠️ Streaming only supports **WAV** or raw **PCM** audio.")
        sample_rate = st.select_slider("Sample rate (Hz)", [8000, 16000, 22050, 48000], value=16000, key="stt_sample_rate")
        uploaded = st.file_uploader("Upload WAV / PCM file", type=["wav"],
                                    key="stt_upload_stream")
        audio_bytes_in = uploaded.read() if uploaded else None
        audio_fname    = uploaded.name   if uploaded else "audio.wav"

    if audio_bytes_in:
        st.audio(audio_bytes_in)

    # ── run ────────────────────────────────────────────────────────────────
    if st.button("▶ Run STT", key="run_stt", use_container_width=True, type="primary"):
        if not audio_bytes_in:
            st.warning("Please upload an audio file first.")
        else:
            client = _require_client()
            log_ph = st.empty()
            log_lines: list[str] = []

            def log_cb(msg: str):
                log_lines.append(msg)
                log_ph.info("\n".join(log_lines[-6:]))

            with st.spinner("Processing…"):
                if stt_path == "REST API":
                    result = stt_module.stt_rest(
                        client, audio_bytes_in, audio_fname, stt_mode, stt_lang,
                    )

                elif stt_path == "Batch API":
                    result = stt_module.stt_batch(
                        client, audio_bytes_in, audio_fname, stt_mode, stt_lang,
                        with_diarization=with_diarization,
                        num_speakers=int(num_speakers) if with_diarization else None,
                        progress_cb=log_cb,
                    )

                else:
                    result = stt_module.stt_streaming_sync(
                        _active_key(), audio_bytes_in, stt_lang, stt_mode,
                        sample_rate=sample_rate, progress_cb=log_cb,
                    )

            log_ph.empty()

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success("✅ Done")
                st.text_area("📝 Transcript", value=result.get("transcript", ""), height=140, key="stt_out")
                if result.get("language_code"):
                    st.caption(f"Detected language: `{result['language_code']}`")
                if result.get("segments"):
                    with st.expander(f"Streaming segments ({len(result['segments'])})"):
                        for i, seg in enumerate(result["segments"], 1):
                            st.write(f"{i}. {seg}")
                if result.get("file_results"):
                    with st.expander("Batch file results"):
                        st.json(result["file_results"])

            _show_raw(result)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Text-to-Speech
# ══════════════════════════════════════════════════════════════════════════════

with col_tts:
    st.header("🔊 Text-to-Speech")

    tts_control = st.radio(
        "Control",
        ["REST — Basic", "REST — Voice", "REST — Advanced", "HTTP Stream", "WebSocket"],
        horizontal=False, key="tts_control",
    )
    st.divider()

    # ── common inputs ──────────────────────────────────────────────────────
    tts_text = st.text_area(
        "Input text",
        value="नमस्ते! मैं सर्वम एआई हूँ। India की हर language को voice देता हूँ।",
        height=90, key="tts_text",
        help="bulbul:v3 supports up to 2500 chars (REST) or 3500 chars (HTTP Stream).",
    )
    tts_lang = st.selectbox("Target language", tts_module.TTS_LANGUAGES, key="tts_lang")

    # ── per-control config ─────────────────────────────────────────────────

    # defaults
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

    if tts_control == "REST — Basic":
        tts_model  = st.selectbox("Model", tts_module.TTS_MODELS, key="tb_model")
        tts_codec  = st.selectbox("Codec", tts_module.REST_CODECS, key="tb_codec")
        tts_speaker = "shubh"

    elif tts_control == "REST — Voice":
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"),
                                   key="tv_speaker")
        tts_model  = st.selectbox("Model", tts_module.TTS_MODELS, key="tv_model")
        tts_codec  = st.selectbox("Codec", tts_module.REST_CODECS, key="tv_codec")

    elif tts_control == "REST — Advanced":
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"),
                                   key="ta_speaker")
        tts_model  = st.selectbox("Model", tts_module.TTS_MODELS, key="ta_model")
        tts_codec  = st.selectbox("Codec", tts_module.REST_CODECS, key="ta_codec")

        st.caption("**Shared**")
        tts_pace = st.slider("Pace", 0.5, 2.0, 1.0, 0.05, key="ta_pace",
                             help="0.5–2.0 (both models)")

        if tts_model == "bulbul:v2":
            st.caption("**bulbul:v2 only**")
            c1, c2 = st.columns(2)
            tts_pitch   = c1.slider("Pitch", -0.75, 0.75, 0.0, 0.05, key="ta_pitch")
            tts_loudness= c2.slider("Loudness", 0.5, 3.0, 1.0, 0.05, key="ta_loudness")
        else:
            st.caption("**bulbul:v3 only**")
            c1, c2 = st.columns(2)
            tts_temp    = c1.slider("Temperature", 0.01, 1.0, 0.6, 0.01, key="ta_temp")
            tts_sr      = c2.select_slider(
                "Sample rate (Hz)", [8000, 16000, 22050, 24000, 32000, 44100, 48000],
                value=22050, key="ta_sr",
            )
            tts_preproc = st.checkbox("Enable preprocessing", False, key="ta_preproc",
                                      help="Normalise English words/numbers before synthesis")

    elif tts_control == "HTTP Stream":
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"),
                                   key="hs_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="hs_model")
        tts_codec   = st.selectbox("Codec", tts_module.STREAM_CODECS, key="hs_codec")
        tts_bitrate = st.selectbox("Bitrate", tts_module.BITRATES, index=3, key="hs_bitrate")
        tts_pace    = st.slider("Pace", 0.5, 2.0, 1.0, 0.05, key="hs_pace")
        if tts_model == "bulbul:v3":
            c1, c2 = st.columns(2)
            tts_temp   = c1.slider("Temperature", 0.01, 1.0, 0.6, 0.01, key="hs_temp")
            tts_preproc = c2.checkbox("Preprocessing", False, key="hs_preproc")

    else:  # WebSocket
        tts_speaker = st.selectbox("Speaker", tts_module.TTS_SPEAKERS,
                                   index=tts_module.TTS_SPEAKERS.index("shubh"),
                                   key="ws_speaker")
        tts_model   = st.selectbox("Model", tts_module.TTS_MODELS, key="ws_model")
        tts_codec   = st.selectbox("Codec", tts_module.WS_CODECS, key="ws_codec")
        tts_bitrate = st.selectbox("Bitrate", tts_module.BITRATES, index=3, key="ws_bitrate")
        c1, c2, c3 = st.columns(3)
        tts_pace    = c1.slider("Pace",    0.5, 2.0,  1.0, 0.05, key="ws_pace")
        tts_pitch   = c2.slider("Pitch", -0.75, 0.75, 0.0, 0.05, key="ws_pitch")
        tts_loudness= c3.slider("Loudness", 0.5, 3.0, 1.0, 0.05, key="ws_loudness")
        tts_sr      = st.select_slider("Sample rate (Hz)",
                                       [8000, 16000, 22050, 24000], value=22050, key="ws_sr")
        tts_preproc = st.checkbox("Enable preprocessing", False, key="ws_preproc")

    # ── run ────────────────────────────────────────────────────────────────
    if st.button("▶ Run TTS", key="run_tts", use_container_width=True, type="primary"):
        if not tts_text.strip():
            st.warning("Please enter some text first.")
        else:
            client  = _require_client()
            log_ph2 = st.empty()
            log2: list[str] = []

            def log_cb2(msg):
                if isinstance(msg, int):
                    log_ph2.caption(f"Received {msg:,} bytes…")
                else:
                    log2.append(msg)
                    log_ph2.info("\n".join(log2[-5:]))

            with st.spinner("Synthesizing…"):
                if tts_control in ("REST — Basic", "REST — Voice", "REST — Advanced"):
                    result = tts_module.tts_rest(
                        client=client,
                        text=tts_text,
                        target_language_code=tts_lang,
                        speaker=tts_speaker,
                        model=tts_model,
                        output_audio_codec=tts_codec,
                        pitch=tts_pitch,
                        loudness=tts_loudness,
                        pace=tts_pace,
                        temperature=tts_temp,
                        speech_sample_rate=tts_sr,
                        enable_preprocessing=tts_preproc,
                    )

                elif tts_control == "HTTP Stream":
                    result = tts_module.tts_http_stream(
                        client=client,
                        text=tts_text,
                        target_language_code=tts_lang,
                        speaker=tts_speaker,
                        model=tts_model,
                        output_audio_codec=tts_codec,
                        output_audio_bitrate=tts_bitrate,
                        pace=tts_pace,
                        temperature=tts_temp,
                        speech_sample_rate=tts_sr,
                        enable_preprocessing=tts_preproc,
                        progress_cb=log_cb2,
                    )

                else:  # WebSocket
                    result = tts_module.tts_websocket_sync(
                        api_key=_active_key(),
                        text=tts_text,
                        target_language_code=tts_lang,
                        speaker=tts_speaker,
                        model=tts_model,
                        output_audio_codec=tts_codec,
                        output_audio_bitrate=tts_bitrate,
                        pace=float(tts_pace or 1.0),
                        pitch=float(tts_pitch or 0.0),
                        loudness=float(tts_loudness or 1.0),
                        speech_sample_rate=int(tts_sr or 22050),
                        enable_preprocessing=bool(tts_preproc),
                        progress_cb=log_cb2,
                    )

            log_ph2.empty()

            if "error" in result:
                st.error(f"❌ {result['error']}")
                if result.get("events"):
                    with st.expander("WebSocket events"):
                        st.json(result["events"])
            else:
                st.success("✅ Audio ready")
                _audio_player(result["audio_bytes"], codec=result.get("codec", "wav"))

                m1, m2, m3 = st.columns(3)
                m1.metric("Audio size", f"{len(result['audio_bytes']):,} B")
                m2.metric("Chunks", result.get("chunks_received", "—"))
                if result.get("events"):
                    m3.metric("WS events", len(result["events"]))

            _show_raw(result)
