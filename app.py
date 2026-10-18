import html
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import main as backend  # the RAG pipeline is called directly; no API layer


@st.cache_resource(show_spinner=False)
def _warm_models() -> bool:
    """Load embedding/reranker/Docling models once per server process."""
    backend.warmup()
    return True


_warm_models()

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnirag_logo.png")
ASSISTANT_AVATAR = LOGO_PATH if os.path.exists(LOGO_PATH) else "🧬"
PAGE_ICON = ASSISTANT_AVATAR


def logo_svg(size: int) -> str:
    """The OmniRag mark: one hub node linked to six satellites (inline SVG)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="{size}" height="{size}"><rect width="100" height="100" rx="24" fill="#D97757"/><line x1="50" y1="50" x2="50.0" y2="23.0" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/><line x1="50" y1="50" x2="73.38" y2="36.5" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/><line x1="50" y1="50" x2="73.38" y2="63.5" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/><line x1="50" y1="50" x2="50.0" y2="77.0" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/><line x1="50" y1="50" x2="26.62" y2="63.5" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/><line x1="50" y1="50" x2="26.62" y2="36.5" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/><line x1="50.0" y1="23.0" x2="73.38" y2="36.5" stroke="#fff" stroke-opacity=".47" stroke-width="1.6" stroke-linecap="round"/><line x1="73.38" y1="36.5" x2="73.38" y2="63.5" stroke="#fff" stroke-opacity=".47" stroke-width="1.6" stroke-linecap="round"/><line x1="73.38" y1="63.5" x2="50.0" y2="77.0" stroke="#fff" stroke-opacity=".47" stroke-width="1.6" stroke-linecap="round"/><line x1="50.0" y1="77.0" x2="26.62" y2="63.5" stroke="#fff" stroke-opacity=".47" stroke-width="1.6" stroke-linecap="round"/><line x1="26.62" y1="63.5" x2="26.62" y2="36.5" stroke="#fff" stroke-opacity=".47" stroke-width="1.6" stroke-linecap="round"/><line x1="26.62" y1="36.5" x2="50.0" y2="23.0" stroke="#fff" stroke-opacity=".47" stroke-width="1.6" stroke-linecap="round"/><circle cx="50.0" cy="23.0" r="5.6" fill="#fff"/><circle cx="73.38" cy="36.5" r="5.6" fill="#fff"/><circle cx="73.38" cy="63.5" r="5.6" fill="#fff"/><circle cx="50.0" cy="77.0" r="5.6" fill="#fff"/><circle cx="26.62" cy="63.5" r="5.6" fill="#fff"/><circle cx="26.62" cy="36.5" r="5.6" fill="#fff"/><circle cx="50" cy="50" r="9" fill="#fff"/></svg>'
    ).replace("{size}", str(size))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniRag",
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded",
)


def md(markup: str, **kwargs):
    """st.markdown wrapper that strips leading whitespace from every line.

    CommonMark treats 4+ leading spaces as an indented code block, which would
    turn unsafe_allow_html HTML into literal text. Dedenting avoids that.
    """
    lines = [line.strip() for line in markup.strip("\n").split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True, **kwargs)


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


# ── Claude-style CSS ─────────────────────────────────────────────────────────
# Streamlit follows the browser/OS theme, so in dark mode it paints text white and
# widgets black. Every colour below is therefore set explicitly (with !important)
# so text is always dark on these light surfaces, whatever the active theme.
md("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
color-scheme: light;
--bg: #FAF9F5;
--panel: #F0EEE6;
--panel-2: #F5F4ED;
--border: #E6E3D8;
--text: #1F1E1D;
--muted: #6B6A63;
--accent: #D97757;
--accent-d: #B8532F;
--good: #2F7D4F;
--warn: #9A6212;
--bad: #B4412F;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
color-scheme: light;
background: var(--bg) !important;
font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 8rem; }

/* ---- default text colour: dark, everywhere ---- */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] em,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stMarkdownContainer"] th,
.stApp [data-testid="stMarkdownContainer"] td,
.stApp [data-testid="stChatMessage"],
.stApp [data-testid="stChatMessageContent"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp label,
.stApp summary,
.stApp details {
color: var(--text) !important;
}
.stApp [data-testid="stMarkdownContainer"] a { color: var(--accent-d) !important; }
.stApp [data-testid="stMarkdownContainer"] code {
background: var(--panel) !important;
color: var(--accent-d) !important;
border-radius: 5px;
padding: 0.1rem 0.35rem;
}
.stApp [data-testid="stMarkdownContainer"] pre { background: var(--panel) !important; border-radius: 10px; }
.stApp [data-testid="stMarkdownContainer"] pre code { color: var(--text) !important; background: transparent !important; }
.stApp [data-testid="stMarkdownContainer"] table { border-color: var(--border) !important; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
background: var(--panel-2) !important;
border-right: 1px solid var(--border);
}
.side-brand {
display: flex; align-items: center; gap: 10px;
font-family: 'Source Serif 4', Georgia, serif;
font-size: 1.35rem; font-weight: 500; letter-spacing: -0.2px;
margin-bottom: 0.9rem;
}
.side-brand svg { border-radius: 8px; }
.section-label {
font-size: 0.68rem; font-weight: 600; letter-spacing: 1.4px;
text-transform: uppercase; margin: 1rem 0 0.4rem 0;
color: var(--muted) !important;
}
.file-chip {
display: flex; justify-content: space-between; gap: 8px;
background: #fff; border: 1px solid var(--border); border-radius: 10px;
padding: 0.4rem 0.7rem; font-size: 0.78rem; margin-bottom: 0.35rem;
}
.file-chip .size { color: var(--muted) !important; white-space: nowrap; }
.side-note { font-size: 0.74rem; line-height: 1.6; color: var(--muted) !important; }
.side-note b { color: var(--text) !important; }

/* ---- file uploader (was black in dark mode) ---- */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
background: #fff !important;
border-color: var(--border) !important;
border-radius: 12px;
}
[data-testid="stFileUploader"] * { color: var(--text) !important; }
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderFileData"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderFile"] { background: var(--panel) !important; border-radius: 10px; }
[data-testid="stFileUploader"] button {
background: #fff !important;
border: 1px solid var(--border) !important;
border-radius: 8px;
}
[data-testid="stFileUploader"] button:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] svg { fill: currentColor; }

/* ---- chat messages ---- */
[data-testid="stChatMessage"] {
background: transparent !important;
border: none;
padding: 0.4rem 0;
gap: 0.9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
background: var(--panel) !important;
border-radius: 16px;
padding: 0.75rem 1rem;
margin-bottom: 0.6rem;
}
[data-testid^="stChatMessageAvatar"] { background: transparent !important; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarAssistant"] img,
[data-testid="stChatMessageAvatarAssistant"] > div { border-radius: 9px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] li {
font-family: 'Source Serif 4', Georgia, serif;
font-size: 1.04rem;
line-height: 1.72;
}

/* ---- composer (was black in dark mode) ---- */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
background: var(--bg) !important;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
background: #fff !important;
color: var(--text) !important;
-webkit-text-fill-color: var(--text) !important;
}
[data-testid="stChatInput"] {
border: 1px solid var(--border) !important;
border-radius: 20px;
box-shadow: 0 2px 14px rgba(31, 30, 29, 0.06);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; }
[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: 1; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 10px; }
[data-testid="stChatInputSubmitButton"] svg { color: #fff !important; fill: #fff !important; }

/* ---- thinking / status / expanders (were dark while running) ---- */
[data-testid="stStatus"], [data-testid="stExpander"],
[data-testid="stStatus"] details, [data-testid="stExpander"] details {
background: #fff !important;
border: 1px solid var(--border) !important;
border-radius: 12px;
}
[data-testid="stStatus"] summary, [data-testid="stExpander"] summary,
[data-testid="stStatus"] summary *, [data-testid="stExpander"] summary * {
color: var(--muted) !important;
font-size: 0.88rem;
}
[data-testid="stStatus"] svg, [data-testid="stExpander"] svg { color: var(--muted) !important; }
.step-row { font-size: 0.84rem; padding: 0.18rem 0; line-height: 1.5; }
.step-row .tick { color: var(--good) !important; margin-right: 6px; }
.step-row .muted { color: var(--muted) !important; }

/* ---- buttons ---- */
.stButton > button {
border-radius: 10px;
border: 1px solid var(--border) !important;
background: #fff !important;
font-weight: 500;
font-size: 0.86rem;
}
.stButton > button, .stButton > button p { color: var(--text) !important; }
.stButton > button:hover { border-color: var(--accent) !important; }
.stButton > button:hover, .stButton > button:hover p { color: var(--accent-d) !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] p { color: #fff !important; }
.stButton > button[kind="primary"]:hover { background: var(--accent-d) !important; }
.stButton > button[kind="primary"]:hover p { color: #fff !important; }

/* ---- pills ---- */
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.6rem 0 0.2rem 0; }
.pill {
display: inline-flex; align-items: center; gap: 5px;
background: var(--panel); border-radius: 999px;
padding: 0.22rem 0.7rem; font-size: 0.74rem; color: var(--muted) !important;
}
.pill b { color: var(--text) !important; font-weight: 600; }
.pill.good { background: #E4F1E8; color: var(--good) !important; }
.pill.bad { background: #F8E3DE; color: var(--bad) !important; }
.pill.src { background: #fff; border: 1px solid var(--border); color: var(--text) !important; }

.notfound-box {
background: var(--panel); border-left: 3px solid var(--warn);
border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.95rem; color: var(--text) !important;
}

/* ---- evaluation cards ---- */
.eval-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.15rem; font-weight: 500; margin: 1rem 0 0.6rem 0; color: var(--text) !important; }
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
@media (max-width: 640px) { .metric-grid { grid-template-columns: 1fr; } }
.metric-card { background: #fff; border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem 1rem; }
.metric-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }
.metric-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 1.1px; text-transform: uppercase; color: var(--muted) !important; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 500; line-height: 1.1; }
.metric-bar { height: 5px; background: var(--panel); border-radius: 999px; margin: 0.55rem 0; overflow: hidden; }
.metric-bar > div { height: 100%; border-radius: 999px; }
.metric-desc { font-size: 0.72rem; line-height: 1.45; color: var(--muted) !important; }
.metric-reason { font-size: 0.78rem; line-height: 1.5; margin-top: 0.55rem; padding-top: 0.55rem; border-top: 1px dashed var(--border); color: var(--text) !important; }
.overall-card { background: var(--panel); border-radius: 14px; padding: 0.8rem 1rem; margin-top: 10px; display: flex; justify-content: space-between; align-items: center; }
.overall-card .name { font-weight: 600; font-size: 0.9rem; color: var(--text) !important; }
.overall-card .num { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; }
.c-good { color: var(--good) !important; } .c-warn { color: var(--warn) !important; } .c-bad { color: var(--bad) !important; }
.bg-good { background: var(--good); } .bg-warn { background: var(--warn); } .bg-bad { background: var(--bad); }

/* ---- empty state ---- */
.hero { text-align: center; padding: 20vh 0 0 0; }
.hero svg { border-radius: 16px; }
.hero .title { font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 400; margin-top: 0.9rem; letter-spacing: -0.4px; color: var(--text) !important; }
.hero .sub { font-size: 0.92rem; margin-top: 0.5rem; color: var(--muted) !important; }
</style>
""")


# ── Labels ───────────────────────────────────────────────────────────────────
STEP_LABELS = {
    "compute_hash_node":              "Fingerprinting your files",
    "load_cache_node":                "Loading cached index",
    "load_documents_node":            "Reading documents",
    "chunking_node":                  "Splitting into chunks",
    "vectorstore_node":               "Embedding into FAISS",
    "save_cache_node":                "Saving index to cache",
    "retrieval_node":                 "Hybrid retrieval (vector + BM25)",
    "rerank_node":                    "Reranking with cross-encoder",
    "filter_chunks_node":             "Grading chunk relevance",
    "check_context_sufficiency_node": "Checking context sufficiency",
    "increase_attempts_node":         "Broadening retrieval",
    "query_rewriting_node":           "Rewriting the search query",
    "generation_node":                "Writing the answer",
    "critic_node":                    "Fact-checking the answer",
    "increase_critic_attempts":       "Retrying after critique",
    "answer_not_found":               "Answer not found",
}

METRICS = [
    ("faithfulness", "Faithfulness",
     "Is every claim in the answer supported by the retrieved context?"),
    ("answer_relevancy", "Answer relevancy",
     "How directly does the answer address the question?"),
    ("contextual_relevancy", "Contextual relevancy",
     "How much of the retrieved context is actually relevant to the question?"),
]


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def start_new_chat():
    st.session_state.messages = []
    st.session_state.uploader_key += 1


# ── Knowledge-base helpers ────────────────────────────────────────

def fetch_collections() -> list[dict]:
    """List persisted collections straight from disk (cheap, so no caching needed)."""
    try:
        return backend.list_knowledge_collections()
    except Exception:
        return []

# ── Rendering helpers ────────────────────────────────────────────────────────
def score_tone(value) -> str:
    if value is None:
        return "bad"
    return "good" if value >= 0.7 else "warn" if value >= 0.4 else "bad"


def step_line(label: str, detail: str, dt: float) -> str:
    detail_html = f' <span class="muted">· {esc(detail)}</span>' if detail else ""
    return (
        f'<div class="step-row"><span class="tick">✓</span>{esc(label)}'
        f'{detail_html} <span class="muted">· {dt:.1f}s</span></div>'
    )


def render_meta_pills(meta: dict):
    pills = []
    pills.append(
        '<span class="pill good">⚡ cached index</span>' if meta.get("from_cache")
        else '<span class="pill">🧠 fresh index</span>'
    )
    pills.append(f'<span class="pill">⏱ <b>{meta.get("elapsed_total", 0):.1f}s</b></span>')
    pills.append(f'<span class="pill">retrievals <b>{meta.get("retrieval_attempts", 0)}</b></span>')
    if meta.get("final_verdict") == "answer_not_found":
        pills.append('<span class="pill bad">not found</span>')
    elif str(meta.get("critic_verdict", "")).lower() == "pass":
        pills.append('<span class="pill good">✓ verified by critic</span>')
    else:
        pills.append('<span class="pill bad">✗ unverified</span>')
    for source in meta.get("sources", []):
        pills.append(f'<span class="pill src">📎 {esc(source)}</span>')
    md(f'<div class="pill-row">{"".join(pills)}</div>')


def render_evaluation(evaluation: dict, elapsed: float | None):
    md('<div class="eval-title">Evaluation</div>')
    cards = []
    for key, label, desc in METRICS:
        res = evaluation.get(key) or {}
        score = res.get("score")
        tone = score_tone(score)
        shown = f"{score:.2f}" if score is not None else "–"
        width = int(max(0.0, min(1.0, score or 0.0)) * 100)
        status = "PASS" if res.get("passed") else "FAIL"
        reason = res.get("reason")
        reason_html = f'<div class="metric-reason">{esc(reason)}</div>' if reason else ""
        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-head"><span class="metric-label">{label}</span>'
            f'<span class="pill {"good" if res.get("passed") else "bad"}">{status}</span></div>'
            f'<div class="metric-value c-{tone}">{shown}</div>'
            f'<div class="metric-bar"><div class="bg-{tone}" style="width:{width}%"></div></div>'
            f'<div class="metric-desc">{desc}</div>{reason_html}</div>'
        )
    md(f'<div class="metric-grid">{"".join(cards)}</div>')

    overall = evaluation.get("overall") or {}
    o_score = overall.get("score")
    o_tone = score_tone(o_score)
    took = f" · {elapsed:.1f}s" if elapsed is not None else ""
    md(
        f'<div class="overall-card"><span class="name">Overall{esc(took)}</span>'
        f'<span class="num c-{o_tone}">{f"{o_score:.2f}" if o_score is not None else "–"}</span></div>'
    )


def run_evaluation(msg: dict):
    """Triggered by the Evaluate button: stream each metric in as it completes."""
    labels = {key: label for key, label, _ in METRICS}
    t0 = time.perf_counter()
    try:
        with st.status("Evaluating answer…", expanded=True) as status:
            rows = {key: st.empty() for key in labels}
            for key, label in labels.items():
                rows[key].markdown(f"⏳ {label}…")

            evaluation, elapsed = {}, None
            for event in backend.stream_evaluation(
                msg["question"], msg["content"], msg["meta"]["evaluation_contexts"]
            ):
                kind = event["type"]
                if kind == "metric":
                    res = event["result"]
                    score = res.get("score")
                    shown = f"{score:.2f}" if score is not None else "failed"
                    rows[event["name"]].markdown(
                        f"✓ {labels[event['name']]} — **{shown}**  ·  {event.get('t', 0):.1f}s"
                    )
                elif kind == "final":
                    evaluation = event["evaluation"]
                    elapsed = event.get("elapsed_total")
                elif kind == "error":
                    raise RuntimeError(event["detail"])

            elapsed = elapsed if elapsed is not None else time.perf_counter() - t0
            status.update(label=f"Evaluation finished in {elapsed:.1f}s", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return

    msg["evaluation"] = {"metrics": evaluation, "elapsed": elapsed}
    st.rerun()


def render_sources(meta: dict):
    """Display retrieved chunks with source IDs, page numbers, and headings."""
    source_chunks = meta.get("source_chunks") or []
    evaluation_contexts = meta.get("evaluation_contexts") or []

    if source_chunks:
        with st.expander(f"📚 Sources ({len(source_chunks)})", expanded=False):
            for source in source_chunks:
                source_id = source.get("citation") or f"[{source.get('id', 'S?')}]"
                filename = source.get("source") or "Unknown source"
                st.markdown(f"**{esc(source_id)} {esc(filename)}**")

                details = []
                pages = source.get("page_numbers") or []
                if pages:
                    page_label = ", ".join(str(page) for page in pages)
                    details.append(f"Page{'s' if len(pages) != 1 else ''} {page_label}")
                headings = source.get("headings") or []
                if headings:
                    details.append("Section: " + " › ".join(str(item) for item in headings))
                if details:
                    st.caption(" · ".join(details))

                content = source.get("content", "")
                if content:
                    st.markdown(content)
                st.divider()
    elif evaluation_contexts:
        # Fallback when only evaluation_contexts is available.
        with st.expander(f"📚 Sources / retrieved context ({len(evaluation_contexts)})", expanded=False):
            for index, context in enumerate(evaluation_contexts, start=1):
                st.markdown(f"**Retrieved chunk {index}**")
                st.markdown(context)
                st.divider()


def render_assistant(index: int, msg: dict):
    meta = msg.get("meta") or {}
    steps = msg.get("steps") or []

    if steps:
        with st.expander(f"Thought for {meta.get('elapsed_total', 0):.1f}s · {len(steps)} steps"):
            md("".join(step_line(s["label"], s["detail"], s["dt"]) for s in steps))

    if meta.get("final_verdict") == "answer_not_found":
        md(f'<div class="notfound-box">⚠ {esc(msg["content"])}</div>')
    else:
        st.markdown(msg["content"])

    if meta:
        render_sources(meta)

    ingestion = meta.get("collection_ingestion") or {}
    added_files = ingestion.get("files_added") or []
    already_present = ingestion.get("files_already_present") or []
    if added_files:
        st.caption(f"Added {len(added_files)} new file(s) to collection **{esc(ingestion.get('name', meta.get('collection_name', '')))}**: " + ", ".join(esc(name) for name in added_files))
    elif already_present:
        st.caption(f"No re-indexing needed — {len(already_present)} file(s) already exist in this collection.")
    skipped_collection_files = ingestion.get("skipped_files") or []
    if skipped_collection_files:
        st.warning("Skipped collection file(s): " + "; ".join(skipped_collection_files))
    if meta.get("skipped_files"):
        st.warning("Skipped unsupported/unreadable file(s): " + ", ".join(meta["skipped_files"]))

    if meta:
        render_meta_pills(meta)

    if msg.get("evaluation"):
        render_evaluation(msg["evaluation"]["metrics"], msg["evaluation"].get("elapsed"))
    elif meta.get("evaluation_contexts") and meta.get("final_verdict") != "answer_not_found":
        if st.button("⚡ Evaluate answer", key=f"eval_btn_{index}", type="primary"):
            run_evaluation(msg)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    md(f'<div class="side-brand">{logo_svg(30)}<span>OmniRag</span></div>')
    if st.button("＋  New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    md('<div class="section-label">Knowledge base</div>')
    available_collections = fetch_collections()
    collection_names = [item.get("name", "") for item in available_collections if item.get("name")]
    new_collection_option = "＋ Create new collection"
    collection_options = collection_names + [new_collection_option]
    default_index = collection_names.index("default") if "default" in collection_names else 0
    selected_collection = st.selectbox(
        "Choose a collection",
        options=collection_options,
        index=default_index,
        help="Collections persist on disk. Upload new files to append them; duplicate file contents are skipped.",
    )
    if selected_collection == new_collection_option:
        collection_name = st.text_input(
            "New collection name", value="default",
            help="Use letters, numbers, spaces, dots, underscores, or hyphens (max 64 characters).",
        ).strip()
    else:
        collection_name = selected_collection
    if collection_names:
        st.caption("Saved collections: " + ", ".join(collection_names))

    md('<div class="section-label">Documents</div>')
    uploaded_files = st.file_uploader(
        label="Upload documents",
        label_visibility="collapsed",
        type=[
            "pdf", "docx", "doc", "odt", "epub",
            "xlsx", "xls", "ods", "csv",
            "pptx", "ppt", "odp",
            "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",
            "html", "xhtml",
            "md", "markdown", "adoc", "asciidoc", "tex", "latex",
            "eml", "msg",
            "wav", "mp3", "m4a", "aac", "ogg", "flac",
            "vtt",
            "json", "xml",
        ],
        help="Add files incrementally to the selected persistent collection. Duplicate content is skipped. Maximum 25 MB per file and 100 MB total per request. CSV uses a row-aware loader; other supported formats use Docling.",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    for f in uploaded_files or []:
        md(f'<div class="file-chip"><span>📎 {esc(f.name)}</span><span class="size">{f.size / 1024:.1f} KB</span></div>')

    md('<div class="section-label">Pipeline</div>')
    md("""
    <div class="side-note">
    Hybrid retrieval (FAISS + BM25) → cross-encoder rerank → relevance grading → grounded answer → critic check.
    Press <b>Evaluate answer</b> under any reply to score it with DeepEval.
    </div>
    """)


# ── Main: composer first (it pins itself to the bottom), then the thread ─────
prompt = st.chat_input("Ask anything about your documents…")

for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            render_assistant(i, message)

if not st.session_state.messages and not prompt:
    hint = (
        "Ask a question below."
        if uploaded_files or collection_name in collection_names else
        "Choose a collection and add documents in the sidebar, then ask away."
    )
    md(f"""
    <div class="hero">
    {logo_svg(60)}
    <div class="title">What's in your documents?</div>
    <div class="sub">{hint}</div>
    </div>
    """)


# ── Handle a new question: stream the pipeline live ──────────────────────────
if prompt:
    question = prompt.strip()

    if not uploaded_files and collection_name not in collection_names:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.warning("Please upload at least one document to create this collection, or select an existing collection.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    files_payload = [(uf.name, uf.getvalue()) for uf in (uploaded_files or [])]

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        status = st.status("Thinking…", expanded=True)
        answer_box = st.empty()

        steps, draft, final = [], "", None
        started = time.perf_counter()

        try:
            for event in backend.stream_query(question, collection_name, files_payload):
                kind = event["type"]

                if kind == "step":
                    label = STEP_LABELS.get(event["node"], event["node"])
                    steps.append({
                        "node": event["node"],
                        "label": label,
                        "detail": event.get("detail", ""),
                        "dt": event.get("dt", 0.0),
                    })
                    status.markdown(step_line(label, event.get("detail", ""), event.get("dt", 0.0)), unsafe_allow_html=True)
                    upcoming = event.get("next")
                    status.update(
                        label=(f"{STEP_LABELS.get(upcoming, upcoming)}…  {event.get('t', 0):.1f}s"
                               if upcoming else f"Wrapping up…  {event.get('t', 0):.1f}s")
                    )

                elif kind == "token":
                    draft += event["text"]
                    answer_box.markdown(draft + " ▌")

                elif kind == "reset":
                    draft = ""
                    answer_box.empty()

                elif kind == "final":
                    final = event

                elif kind == "error":
                    raise RuntimeError(event["detail"])

            if final is None:
                raise RuntimeError("The pipeline ended without a result.")

            elapsed = float(final.get("elapsed_total", time.perf_counter() - started))
            status.update(
                label=f"Thought for {elapsed:.1f}s · {len(steps)} steps",
                state="complete",
                expanded=False,
            )
            answer_box.empty()

        except Exception as exc:
            status.update(label="Pipeline error", state="error")
            st.error(f"Pipeline error: {exc}")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": final.get("answer", "No answer returned."),
        "question": question,
        "steps": steps,
        "meta": final,
        "evaluation": None,
    })
    st.rerun()
