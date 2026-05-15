"""
Arabic Audio Intelligence — LOCAL MODELS version
=================================================
Models:
  - Whisper small         ~461 MB   ASR
  - AraBART summarization ~500 MB   Summarization
  - ArabERT v02           ~500 MB   Semantic Search

Total download: ~1.5 GB

Requirements:
  pip install streamlit pytubefix torch transformers sentencepiece
  pip install protobuf accelerate faiss-cpu openai-whisper
"""

import streamlit as st
import os
import time
import tempfile
import numpy as np
import torch

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Audio Intelligence",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary:    #0a0a0f;
    --bg-secondary:  #111118;
    --bg-card:       #16161f;
    --bg-input:      #1c1c28;
    --accent-gold:   #c9a84c;
    --accent-teal:   #2dd4bf;
    --accent-rose:   #f43f5e;
    --accent-violet: #8b5cf6;
    --text-primary:  #f0ede6;
    --text-muted:    #6b7280;
    --border:        #2a2a3a;
}

html, body, [class*="css"] {
    font-family: 'Tajawal', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.stDeployButton { display: none; }

.main .block-container { padding: 2rem 1.5rem 4rem; max-width: 860px; }

.hero { text-align: center; padding: 2.5rem 0 2rem; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.3em; color: var(--accent-gold);
    text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.6rem; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #f0ede6 0%, #c9a84c 50%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}
.hero-sub { font-size: 1rem; color: var(--text-muted); font-weight: 300; direction: rtl; }

.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.25rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; opacity: 0.6;
}
.card-gold::before   { background: linear-gradient(90deg, transparent, #c9a84c, transparent); }
.card-teal::before   { background: linear-gradient(90deg, transparent, #2dd4bf, transparent); }
.card-violet::before { background: linear-gradient(90deg, transparent, #8b5cf6, transparent); }

.card-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.75rem;
}
.label-gold   { color: var(--accent-gold); }
.label-teal   { color: var(--accent-teal); }
.label-violet { color: var(--accent-violet); }

.metrics-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 100px; background: var(--bg-input);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 0.9rem; text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace; font-size: 1.6rem;
    font-weight: 600; line-height: 1; margin-bottom: 0.3rem;
}
.metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }

.segment {
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.55rem;
    display: flex; gap: 1rem; align-items: flex-start;
}
.segment-time {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    color: var(--accent-gold); white-space: nowrap; padding-top: 2px; min-width: 90px;
}
.segment-text { font-size: 0.93rem; line-height: 1.6; direction: rtl; text-align: right; flex: 1; }

.summary-box {
    background: var(--bg-input); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.25rem 1.5rem; font-size: 1rem; line-height: 1.9;
    direction: rtl; text-align: right; color: var(--text-primary);
}

.search-hit {
    background: var(--bg-input); border: 1px solid var(--border);
    border-left: 3px solid var(--accent-teal); border-radius: 10px;
    padding: 1rem 1.25rem; margin-bottom: 0.75rem;
}
.search-hit-meta {
    display: flex; justify-content: space-between; margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
}
.search-score { color: var(--accent-teal); }
.search-ts    { color: var(--accent-gold); }
.search-text  { font-size: 0.93rem; direction: rtl; text-align: right; line-height: 1.6; }

.stTextInput > div > div > input {
    background: var(--bg-input) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
    font-family: 'Tajawal', sans-serif !important; font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.25) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold), #a07830) !important;
    color: #0a0a0f !important; border: none !important; border-radius: 8px !important;
    font-family: 'Tajawal', sans-serif !important; font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important; width: 100% !important;
    font-size: 1rem !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stDownloadButton > button {
    background: var(--bg-input) !important; color: var(--accent-teal) !important;
    border: 1px solid var(--accent-teal) !important; border-radius: 8px !important;
    font-family: 'Tajawal', sans-serif !important; width: auto !important;
}
.stProgress > div > div { background: var(--accent-gold) !important; }
.stAlert { border-radius: 10px !important; }
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important; border-radius: 10px !important;
    padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 7px !important;
    color: var(--text-muted) !important; font-family: 'Tajawal', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important; color: var(--accent-gold) !important;
}
.stTextArea > div > div > textarea {
    background: var(--bg-input) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
    font-family: 'Tajawal', sans-serif !important; direction: rtl;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────
for k, v in {
    "segments": [], "full_transcript": "",
    "summary": "", "srt_content": "", "search_engine": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Whisper small (~461 MB, one-time download)…")
def load_whisper():
    import whisper
    return whisper.load_model("small", device=DEVICE)


@st.cache_resource(show_spinner="Loading AraBART summarizer (~500 MB, one-time download)…")
def load_summarizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "malmarjeh/mbert2mbert-arabic-text-summarization"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
    mdl.eval()
    return tok, mdl


@st.cache_resource(show_spinner="Loading ArabERT embedder (~500 MB, one-time download)…")
def load_embedder():
    from transformers import AutoTokenizer, AutoModel
    name = "aubmindlab/bert-base-arabertv02"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModel.from_pretrained(name).to(DEVICE)
    mdl.eval()
    return tok, mdl


# ══════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def seconds_to_srt(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    sc   = int(s % 60)
    ms   = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

def seconds_to_ts(s):
    return f"{int(s//60):02d}:{int(s%60):02d}"

def generate_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i),
                  f"{seconds_to_srt(seg['start'])} --> {seconds_to_srt(seg['end'])}",
                  seg["text"], ""]
    return "\n".join(lines)


def transcribe_audio(audio_path: str):
    model  = load_whisper()
    result = model.transcribe(
        audio_path,
        language="ar",
        verbose=False,
        word_timestamps=False,
    )
    full_text = result["text"].strip()
    segments  = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
        if seg["text"].strip()
    ]
    return segments, full_text


def summarize_text(text: str) -> str:
    tok, mdl  = load_summarizer()
    words     = text.split()
    # AraBART max input ~1024 tokens — keep chunks safe at ~300 words
    chunks    = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
    summaries = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        # No "summarize: " prefix for AraBART
        input_ids = tok(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).input_ids.to(DEVICE)
        with torch.no_grad():
            output_ids = mdl.generate(
                input_ids,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        summaries.append(tok.decode(output_ids[0], skip_special_tokens=True))
    return " ".join(summaries)


def get_embeddings(texts: list) -> np.ndarray:
    tok, mdl = load_embedder()
    all_embs = []
    # Batch in groups of 16 to avoid OOM on CPU
    for i in range(0, len(texts), 16):
        batch   = texts[i:i+16]
        encoded = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = mdl(**encoded)
        # Mean pool over token dimension
        embs = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
        all_embs.append(embs)
    embs  = np.vstack(all_embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms == 0, 1, norms)


def build_search_index(segments: list):
    import faiss
    texts = [s["text"] for s in segments]
    embs  = get_embeddings(texts)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def search_segments(query: str, segments: list, index, top_k: int = 5):
    q_emb      = get_embeddings([query])
    scores, ix = index.search(q_emb, top_k)
    return [
        {"score": float(s), **segments[i]}
        for s, i in zip(scores[0], ix[0])
        if i < len(segments)
    ]


def download_yt_audio(url: str):
    from pytubefix import YouTube
    yt           = YouTube(url)
    title        = yt.title or "Unknown"
    audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if not audio_stream:
        raise RuntimeError("No audio stream found for this video.")
    tmp      = tempfile.mkdtemp()
    out_path = audio_stream.download(output_path=tmp, filename="audio")
    return out_path, title


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Deep Learning · NLP · Arabic — Local Models</div>
    <div class="hero-title">Arabic Audio Intelligence</div>
    <div class="hero-sub">تحويل الصوت إلى نص · التلخيص · البحث الدلالي</div>
</div>
""", unsafe_allow_html=True)

device_color = "#2dd4bf" if DEVICE == "cuda" else "#c9a84c"
st.markdown(
    f'<div style="text-align:center;margin-bottom:1rem">'
    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
    f'color:{device_color};background:var(--bg-card);padding:4px 12px;'
    f'border-radius:20px;border:1px solid var(--border)">⚡ Running on {DEVICE.upper()}</span></div>',
    unsafe_allow_html=True,
)

# ── URL input ─────────────────────────────────────────────────────
st.markdown('<div class="card card-gold"><div class="card-label label-gold">YouTube URL</div>', unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])
with col_input:
    url = st.text_input("url", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
with col_btn:
    run = st.button("🚀 Go")
st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["segments"]:
    st.markdown(
        '<div style="text-align:center;font-size:0.78rem;color:var(--text-muted);margin-bottom:1rem">'
        '⏱ First run downloads ~1.5 GB · CPU: ~6–10 min per video · GPU: ~1 min</div>',
        unsafe_allow_html=True,
    )

# ── Run pipeline ──────────────────────────────────────────────────
if run and url:
    prog = st.progress(0, text="Downloading audio…")
    try:
        audio_path, title = download_yt_audio(url)
        st.info(f"🎬 **{title}**")

        prog.progress(10, text="Loading Whisper…")
        load_whisper()

        prog.progress(20, text="Transcribing with Whisper-small…")
        t0 = time.time()
        segs, full_text = transcribe_audio(audio_path)
        st.caption(f"✅ Transcription: {time.time()-t0:.1f}s")

        prog.progress(55, text="Loading AraBART summarizer…")
        load_summarizer()

        prog.progress(60, text="Generating Arabic summary…")
        t0      = time.time()
        summary = summarize_text(full_text) if full_text.strip() else "لا يوجد نص للتلخيص."
        st.caption(f"✅ Summarization: {time.time()-t0:.1f}s")

        prog.progress(80, text="Loading ArabERT & building search index…")
        load_embedder()
        t0    = time.time()
        index = build_search_index(segs)
        srt   = generate_srt(segs)
        st.caption(f"✅ Search index: {time.time()-t0:.1f}s")

        prog.progress(100, text="Done ✅")
        st.session_state.update({
            "segments":        segs,
            "full_transcript": full_text,
            "summary":         summary,
            "srt_content":     srt,
            "search_engine":   (index, segs),
        })
        try:
            os.remove(audio_path)
        except Exception:
            pass
        st.success(f"✅ Done — {len(segs)} segments · {len(full_text.split())} words")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ── Results ───────────────────────────────────────────────────────
if st.session_state["segments"]:
    segs  = st.session_state["segments"]
    dur   = segs[-1]["end"] if segs else 0
    words = len(st.session_state["full_transcript"].split())

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-box">
            <div class="metric-value" style="color:var(--accent-gold)">{len(segs)}</div>
            <div class="metric-label">Segments</div>
        </div>
        <div class="metric-box">
            <div class="metric-value" style="color:var(--accent-teal)">{int(dur//60)}m {int(dur%60)}s</div>
            <div class="metric-label">Duration</div>
        </div>
        <div class="metric-box">
            <div class="metric-value" style="color:var(--accent-violet)">{words:,}</div>
            <div class="metric-label">Words</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_trans, tab_sum, tab_search = st.tabs(["📋 Transcript", "📝 Summary", "🔍 Search"])

    with tab_trans:
        st.markdown('<div class="card card-gold"><div class="card-label label-gold">Timestamped Segments</div>', unsafe_allow_html=True)
        for seg in segs:
            st.markdown(f"""
            <div class="segment">
                <div class="segment-time">{seconds_to_ts(seg['start'])} → {seconds_to_ts(seg['end'])}</div>
                <div class="segment-text">{seg['text']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download SRT", st.session_state["srt_content"],
                           file_name="arabic_subtitles.srt", mime="text/plain")
        st.download_button("⬇️ Download Full Text", st.session_state["full_transcript"],
                           file_name="transcript.txt", mime="text/plain")

    with tab_sum:
        st.markdown(
            f'<div class="card card-teal"><div class="card-label label-teal">Auto-generated Summary · AraBART</div>'
            f'<div class="summary-box" style="margin-top:.75rem">{st.session_state["summary"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.download_button("⬇️ Download Summary", st.session_state["summary"],
                           file_name="summary.txt", mime="text/plain")

    with tab_search:
        st.markdown('<div class="card card-violet"><div class="card-label label-violet">Semantic Search · ArabERT</div>', unsafe_allow_html=True)
        query     = st.text_input("Search query", placeholder="ما هو الموضوع الرئيسي؟", label_visibility="collapsed")
        do_search = st.button("🔍 Search", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if do_search and query and st.session_state["search_engine"]:
            index, s_segs = st.session_state["search_engine"]
            t0      = time.time()
            results = search_segments(query, s_segs, index, top_k=5)
            elapsed = (time.time() - t0) * 1000

            st.markdown(
                f'<div style="font-size:.78rem;color:var(--text-muted);margin:1rem 0 .5rem">'
                f'Found <strong style="color:var(--accent-violet)">{len(results)}</strong> results in '
                f'<strong style="color:var(--accent-gold)">{elapsed:.1f}ms</strong></div>',
                unsafe_allow_html=True,
            )

            for r in results:
                pct       = max(0, min(100, int(r["score"] * 100)))
                bar_color = "#8b5cf6" if pct > 70 else "#c9a84c" if pct > 50 else "#6b7280"
                st.markdown(f"""
                <div class="search-hit">
                    <div class="search-hit-meta">
                        <span class="search-ts">⏱ {seconds_to_ts(r['start'])} → {seconds_to_ts(r['end'])}</span>
                        <span class="search-score">Score: {pct}%</span>
                    </div>
                    <div style="background:var(--bg-primary);border-radius:4px;height:4px;margin-bottom:.6rem">
                        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px"></div>
                    </div>
                    <div class="search-text">{r['text']}</div>
                </div>
                """, unsafe_allow_html=True)