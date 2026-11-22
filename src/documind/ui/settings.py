"""Settings (model, key, retrieval, storage, appearance) and About pages."""

from __future__ import annotations

from html import escape

import streamlit as st

from documind import __version__
from documind.app_factory import DocuMindApp
from documind.ui import session
from documind.ui.theme import TOKENS, dot, html, notice, page_header, theme_type


def _kv(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    html(f'<dl class="dm-kv">{body}</dl>')


def render_settings(app: DocuMindApp) -> None:
    s = app.settings
    page_header("Settings", "Model connection, retrieval parameters and storage.")

    # ---- Model
    st.markdown("#### Model")
    status = session.model_status(app)
    html(f'<div class="dm-status" style="padding-left:0">{dot(status.kind)}<b>{escape(status.label)}</b></div>')
    st.caption(status.detail)
    emb = app.embedder
    _kv(
        [
            ("Answer generation", s.gemini_model if app.generation_enabled else "Off (no API key)"),
            (
                "Embeddings",
                f"{emb.model}, {emb.dimension} dimensions"
                if emb.provider == "gemini"
                else "Offline lexical (feature hashing, no API). Word overlap only.",
            ),
            ("Vector space", emb.key),
        ]
    )
    if st.button(
        "Test connection",
        icon=":material/wifi_tethering:",
        disabled=not app.generation_enabled,
        help="Checks the key and both models with the free models.get endpoint. No tokens are used.",
    ):
        with st.spinner("Contacting Gemini…"):
            session.test_connection(app)
        st.rerun()

    # ---- API key
    st.markdown("#### API key")
    if s.gemini_api_key:
        st.caption(
            "Loaded on the server from `.env` or Streamlit secrets. It is never displayed, logged or sent to "
            "the browser."
        )
    else:
        st.caption(
            "Recommended: put `GEMINI_API_KEY` in `.env` (local) or Streamlit secrets (hosted) and restart. "
            "Get a key at aistudio.google.com/apikey."
        )
        with st.form("api_key_form", border=False):
            key = st.text_input(
                "Use a key for this browser session only", type="password", placeholder="Paste a Gemini API key"
            )
            c1, _ = st.columns([1, 3])
            submitted = c1.form_submit_button("Use key", type="primary")
        if submitted and key.strip():
            st.session_state.user_api_key = key.strip()
            st.session_state.gemini_check = None
            st.rerun()
        if st.session_state.user_api_key:
            notice("A session key is active.", "accent")
            if st.button("Forget session key", icon=":material/key_off:"):
                st.session_state.user_api_key = ""
                st.session_state.pop("_user_app", None)
                st.session_state.gemini_check = None
                st.rerun()
        st.caption(
            "A session key is kept in this session's server-side memory only: it is not written to disk, logs, "
            "URLs or browser storage, and it is dropped when the session ends. Whoever runs the server process "
            "could still read its memory, so use your own server for real keys."
        )

    # ---- Retrieval
    st.markdown("#### Retrieval")
    # Widget state is discarded on other pages, so keep the durable value under "top_k".
    st.session_state.top_k_widget = st.session_state.top_k
    st.slider(
        "Passages retrieved per question (top-k)",
        1,
        15,
        key="top_k_widget",
        on_change=lambda: st.session_state.update(top_k=st.session_state.top_k_widget),
        help="More passages give more context but longer prompts.",
    )
    _kv(
        [
            ("Chunk size", f"{s.chunk_size} characters, {s.chunk_overlap} overlap"),
            ("Minimum similarity", f"{s.min_similarity:.2f} (cosine; 0 keeps every top-k result)"),
            ("Context budget", f"{s.max_context_chars:,} characters"),
            ("Question limit", f"{s.max_question_chars:,} characters"),
        ]
    )

    # ---- Storage
    st.markdown("#### Storage")
    records = app.ingestion.list_documents()
    _kv(
        [
            ("Mode", "Private temporary store per browser session" if s.session_isolation else "Local disk"),
            ("Location", str(session.persist_dir())),
            ("Documents", f"{len(records)} ({len(app.ingestion.searchable_ids())} searchable)"),
            ("Upload limit", f"{s.max_upload_mb} MB per file, PDF and DOCX"),
        ]
    )

    # ---- Appearance
    st.markdown("#### Appearance")
    st.caption(
        f"Current theme: {theme_type()}. Dark is the default. To switch, open the ⋮ menu at the top right, "
        "choose Settings, then Theme."
    )
    st.caption(f"DocuMind {__version__}")


def render_about(app: DocuMindApp) -> None:
    page_header("About DocuMind", "Answers from your documents, with the evidence attached.")
    st.markdown(
        """
DocuMind is a retrieval-augmented generation (RAG) application. It answers only from the documents you
select, and every claim links back to the page or section it came from, so you can check it.
"""
    )
    st.markdown("#### How a question is answered")
    t = TOKENS[theme_type()]
    st.graphviz_chart(
        f"""
digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded", color="{t["border"]}", fontcolor="{t["text"]}", fontname="Helvetica",
        fontsize=10, penwidth=1.2, margin="0.12,0.06"];
  edge [color="{t["muted"]}", arrowsize=0.6];
  Q [label="Question"]; E [label="Embed"]; R [label="Search selected\\ndocuments"];
  P [label="Grounded prompt\\n(sources S1..Sn)"]; G [label="Gemini", color="{t["accent"]}"];
  V [label="Validate\\ncitations"]; A [label="Answer +\\nevidence"];
  Q -> E -> R -> P -> G -> V -> A;
}}
""",
        width="stretch",
    )
    st.markdown(
        """
#### Design decisions

1. **Exact locations.** Chunks never cross a PDF page or a Word section, so each citation points to one place.
2. **Citations the model cannot invent.** The model only sees IDs `S1…Sn` issued by the app; anything else is
   removed before display.
3. **Documents are untrusted.** Passages are fenced and escaped, and the model is told never to follow
   instructions inside them.
4. **Strict scope.** Search is filtered by document ID in ChromaDB and checked again in Python.
5. **Versioned vectors.** Each embedding model has its own collection; switching models requires a re-index.

#### Stack

Python 3.12, Streamlit, Google Gemini (`google-genai`), ChromaDB, PyMuPDF, python-docx, Pydantic, Pytest, Ruff.

#### Limitations

- No OCR: scanned PDFs are detected and flagged, not read.
- No user accounts. Run locally, or enable session isolation before hosting.
- The offline lexical embedder matches words, not meaning, and cannot match Vietnamese questions to English text.
- Similarity scores are cosine similarities, not confidence.
- Grounding reduces but does not eliminate model errors. Check the cited evidence.
"""
    )
