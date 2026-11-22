"""Settings (answer generation, embeddings, retrieval, limits, storage) and About pages."""

from __future__ import annotations

from html import escape

import streamlit as st

from documind import __version__
from documind.app_factory import DocuMindApp
from documind.llm.shopaikey import PROTOCOLS
from documind.ui import session
from documind.ui.theme import TOKENS, dot, html, notice, page_header, theme_type

_PROVIDERS = ["gemini", "shopaikey", "offline"]


def _kv(rows: list[tuple[str, str]]) -> None:
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    html(f'<dl class="dm-kv">{body}</dl>')


def _bind(widget_key: str, durable_key: str) -> None:
    """Copy a widget value into durable session state (widget state is discarded off-page)."""
    value = st.session_state.get(widget_key)
    if value is not None:
        st.session_state[durable_key] = value.strip() if isinstance(value, str) else value


def _status_line(app: DocuMindApp) -> None:
    status = session.model_status(app)
    text = session.STATE_TEXT.get(status.state, status.state)
    html(
        f'<div class="dm-status" style="padding-left:0">{dot(status.kind)}'
        f"<b>{escape(status.label)}</b><span>{escape(text)}</span></div>"
    )
    st.caption(status.detail)


def _session_key_form(form_key: str, label: str, placeholder: str, durable_key: str) -> None:
    """Password field; the value moves to server-side session state and the form is cleared."""
    with st.form(form_key, border=False, clear_on_submit=True):
        key = st.text_input(label, type="password", placeholder=placeholder, autocomplete="off")
        submitted = st.form_submit_button("Use key", type="primary")
    if submitted and key.strip():
        st.session_state[durable_key] = key.strip()
        st.rerun()
    if st.session_state.get(durable_key):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.caption("A session key is active (hidden).")
        if c2.button("Forget key", icon=":material/key_off:", key=f"forget_{form_key}", width="stretch"):
            st.session_state[durable_key] = ""
            st.rerun()


def _test_button(app: DocuMindApp, help_text: str) -> None:
    if st.button(
        "Test connection", icon=":material/wifi_tethering:", disabled=not app.generation_enabled, help=help_text
    ):
        with st.spinner("Sending one short request…"):
            session.test_connection(app)
        st.rerun()


def _gemini_section(app: DocuMindApp) -> None:
    s = app.settings
    _kv([("Model", s.gemini_model), ("Endpoint", "Google Gemini API (generativelanguage.googleapis.com)")])
    if s.gemini_api_key:
        st.caption("Key loaded on the server from `.env` or Streamlit secrets. It is never displayed or logged.")
    else:
        st.caption("Put `GEMINI_API_KEY` in `.env` (local) or Streamlit secrets (hosted), or use a session key.")
        _session_key_form(
            "gemini_key_form", "Gemini API key for this session", "Paste a Gemini API key", "user_api_key"
        )
    _test_button(app, "Checks the key and model with the free models.get endpoint. No tokens are used.")


def _model_picker(protocol: str, has_key: bool) -> None:
    st.markdown("###### Model")
    catalog = session.model_catalog()
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    if catalog and catalog["models"]:
        options = catalog["models"]
        current = st.session_state.shopaikey_model
        st.session_state.sak_model_pick = current if current in options else None
        c1.selectbox(
            f"Models in the gateway catalog ({len(options)})",
            options,
            key="sak_model_pick",
            index=None,
            placeholder="Choose a model",
            on_change=_bind,
            args=("sak_model_pick", "shopaikey_model"),
        )
    else:
        c1.caption("Load the catalog to pick a model, or type a model ID below.")
    if c2.button(
        "Load models",
        icon=":material/refresh:",
        disabled=not has_key,
        width="stretch",
        help="One request to the documented model catalog endpoint.",
    ):
        with st.spinner("Loading the model catalog…"):
            session.refresh_model_catalog()
        st.rerun()
    if catalog and catalog.get("error"):
        notice(catalog["error"], "err")
    elif catalog:
        extra = (
            " No Anthropic-format catalog is documented, so Claude models from the gateway catalog are shown."
            if protocol == "anthropic"
            else ""
        )
        st.caption(
            "This is the gateway catalog. A key can only use models in its model group, so test the model you "
            "choose." + extra
        )
    st.session_state.sak_model_text = st.session_state.shopaikey_model
    st.text_input(
        "Model ID",
        key="sak_model_text",
        placeholder="Exact model ID, e.g. from the catalog",
        on_change=_bind,
        args=("sak_model_text", "shopaikey_model"),
        help="Case-sensitive, exactly as ShopAIKey lists it.",
    )


def _shopaikey_section(app: DocuMindApp) -> None:
    s = app.settings
    st.caption(
        "ShopAIKey is a third-party API gateway. With it enabled, your question and the retrieved document "
        "excerpts are sent to the gateway and on to the model vendor, so they leave this machine. Usage is "
        "billed to your ShopAIKey credit."
    )
    st.session_state.sak_protocol_widget = st.session_state.shopaikey_protocol
    st.segmented_control(
        "API protocol",
        list(PROTOCOLS),
        format_func=lambda p: PROTOCOLS[p],
        key="sak_protocol_widget",
        on_change=_bind,
        args=("sak_protocol_widget", "shopaikey_protocol"),
        required=True,
    )
    protocol = st.session_state.shopaikey_protocol
    if s.shopaikey_api_key:
        st.caption("ShopAIKey key loaded on the server from `.env` or Streamlit secrets (hidden).")
    else:
        _session_key_form("shopaikey_key_form", "ShopAIKey API key for this session", "sk-…", "shopaikey_key")
    _model_picker(protocol, has_key=bool(session.shopaikey_key()))
    root = s.shopaikey_base_url.rstrip("/").removesuffix("/v1")
    endpoint = f"{root}/v1" if protocol == "openai" else root
    _kv([("Endpoint", f"{endpoint} ({PROTOCOLS[protocol]})")])
    _test_button(app, "Sends one very short generation request (a few tokens) to verify key, model and group.")


def _generation_section(app: DocuMindApp) -> None:
    st.markdown("#### Answer generation")
    st.session_state.provider_widget = st.session_state.llm_provider
    st.segmented_control(
        "AI provider",
        _PROVIDERS,
        format_func=lambda p: session.PROVIDER_LABELS[p],
        key="provider_widget",
        on_change=_bind,
        args=("provider_widget", "llm_provider"),
        required=True,
    )
    _status_line(app)
    provider = st.session_state.llm_provider
    if provider == "gemini":
        _gemini_section(app)
    elif provider == "shopaikey":
        _shopaikey_section(app)


def _embedding_section(app: DocuMindApp) -> None:
    s = app.settings
    emb = app.embedder
    st.markdown("#### Retrieval and embeddings")
    if emb.provider == "local":
        kind = "Offline lexical index (feature hashing). Matches words, not meaning; not a semantic embedding."
    elif emb.provider == "gemini":
        kind = f"Gemini embeddings: {emb.model}, {emb.dimension} dimensions"
    else:
        kind = f"ShopAIKey embeddings (OpenAI format): {emb.model}, {emb.dimension} dimensions"
    _kv([("Search mode", kind), ("Vector space", emb.key)])
    for text in app.notices:
        notice(text, "warn")
    st.caption(
        "Embeddings are set on the server (`DOCUMIND_EMBEDDING_PROVIDER`) and are independent of the answer "
        "provider. Changing them creates a new vector space; re-index documents on the Documents page."
    )
    if emb.provider != "local" and st.button(
        "Test embedding endpoint", icon=":material/wifi_tethering:", help="Embeds one short word."
    ):
        ok, message = session.test_embeddings(app)
        notice(message, "ok" if ok else "err")

    # Widget state is discarded on other pages, so keep the durable value under "top_k".
    st.session_state.top_k_widget = st.session_state.top_k
    st.slider(
        "Passages retrieved per question (top-k)",
        1,
        15,
        key="top_k_widget",
        on_change=lambda: st.session_state.update(top_k=st.session_state.top_k_widget),
        help="More passages give more context but longer (more expensive) prompts.",
    )
    _kv(
        [
            ("Chunk size", f"{s.chunk_size} characters, {s.chunk_overlap} overlap"),
            ("Minimum similarity", f"{s.min_similarity:.2f} (cosine; 0 keeps every top-k result)"),
            ("Context budget", f"{s.max_context_chars:,} characters"),
        ]
    )


def render_settings(app: DocuMindApp) -> None:
    s = app.settings
    page_header("Settings", "Answer provider, retrieval, limits and storage.")
    _generation_section(app)
    _embedding_section(app)

    st.markdown("#### Limits")
    _kv(
        [
            ("Max output tokens", f"{s.max_output_tokens:,} per answer (DOCUMIND_MAX_OUTPUT_TOKENS)"),
            ("Request timeout", f"{s.request_timeout_s:.0f} s"),
            ("Retries", f"at most {s.max_retries}, transient errors only (never for balance or model errors)"),
            ("Question limit", f"{s.max_question_chars:,} characters"),
        ]
    )
    st.caption("Token usage is shown under each answer when the provider reports it. No prices are estimated.")

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
  P [label="Grounded prompt\\n(sources S1..Sn)"]; G [label="LLM provider\\n(Gemini or ShopAIKey)", color="{t["accent"]}"];
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
6. **Independent providers.** Answer generation (Gemini or ShopAIKey) and embeddings are configured separately.

#### Stack

Python 3.12, Streamlit, Google Gemini (`google-genai`), ShopAIKey gateway (`openai`, `anthropic`, `google-genai`
SDKs), ChromaDB, PyMuPDF, python-docx, Pydantic, Pytest, Ruff.

#### Limitations

- No OCR: scanned PDFs are detected and flagged, not read.
- No user accounts. Run locally, or enable session isolation before hosting.
- The offline lexical embedder matches words, not meaning, and cannot match Vietnamese questions to English text.
- Similarity scores are cosine similarities, not confidence.
- Grounding reduces but does not eliminate model errors. Check the cited evidence.
"""
    )
