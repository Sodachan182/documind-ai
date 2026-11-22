"""Session state, app construction and provider status for the Streamlit UI.

Durable state lives under plain keys in ``st.session_state``; widgets use their
own keys (Streamlit discards widget state on pages where the widget is not
rendered) and are synced from/to the durable keys.

Keys typed into the UI are held only in this session's server-side state: they
are not written to disk, logs, URLs, browser storage or exports, and they are
never placed in a process-wide cache shared with other sessions.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from documind.app_factory import DocuMindApp, GenerationSelection, build_app, build_generator, with_generator
from documind.config import SECRET_ENV_NAMES, Settings, load_settings
from documind.embeddings.base import EmbeddingError
from documind.llm.errors import GenerationError
from documind.llm.shopaikey import PROTOCOLS
from documind.llm.shopaikey import list_models as list_shopaikey_models
from documind.models import RAGAnswer

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "demo" / "documents"
QUESTIONS_FILE = ROOT / "demo" / "questions.json"
RESULTS_DIR = ROOT / "docs" / "evaluation_results"
_SECRET_KEYS = (
    *SECRET_ENV_NAMES,
    "DOCUMIND_LLM_PROVIDER",
    "GEMINI_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "SHOPAIKEY_MODEL",
    "SHOPAIKEY_PROTOCOL",
    "SHOPAIKEY_BASE_URL",
    "DOCUMIND_EMBEDDING_PROVIDER",
    "DOCUMIND_EMBEDDING_MODEL",
    "DOCUMIND_EMBEDDING_DIMENSION",
    "DOCUMIND_SESSION_ISOLATION",
)

OFFLINE_DETAIL = "No model is called. Chat shows the most relevant passages only; nothing leaves this machine."
PROVIDER_LABELS = {"gemini": "Google Gemini", "shopaikey": "ShopAIKey", "offline": "Offline"}
STATE_TEXT = {
    "not_configured": "Not configured",
    "ready": "Ready to test",
    "connected": "Connected",
    "auth_failed": "Authentication failed",
    "model_unavailable": "Model unavailable",
    "rate_limited": "Rate limited",
    "insufficient_balance": "Insufficient balance",
    "timeout": "Timed out",
    "service_error": "Service unavailable",
    "connection_error": "Connection failed",
    "bad_request": "Request rejected",
    "error": "Error",
    "offline": "Retrieval only",
}
_STATE_KIND = {"connected": "ok", "ready": "accent", "offline": "muted", "not_configured": "warn"}


# --------------------------------------------------------------------------- settings / app
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    try:  # Streamlit Cloud keeps secrets in st.secrets; mirror known keys into env vars.
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    for key in _SECRET_KEYS:
        if key in secrets and not os.environ.get(key):
            os.environ[key] = str(secrets[key])
    return load_settings(ROOT / ".env")


@st.cache_resource(show_spinner="Opening knowledge base…")
def _shared_app(persist_dir: str) -> DocuMindApp:
    """Knowledge base + retriever shared by sessions using the same store.

    Built without a generator: each session attaches its own (see ``get_app``).
    """
    return build_app(get_settings(), persist_dir=Path(persist_dir), selection=GenerationSelection("offline"))


def init_state() -> None:
    settings = get_settings()
    defaults: dict[str, Any] = {
        "session_id": uuid.uuid4().hex,
        "conversations": {},
        "current_conv": None,
        "user_api_key": "",  # session-only Gemini key
        "shopaikey_key": "",  # session-only ShopAIKey key
        "llm_provider": settings.resolved_llm_provider(),
        "shopaikey_protocol": settings.shopaikey_protocol,
        "shopaikey_model": settings.shopaikey_model,
        "model_catalog": {},
        "conn": {},
        "doc_scope": None,
        "known_docs": [],
        "uploader_key": 0,
        "top_k": settings.top_k,
        "processed": set(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_dir() -> Path:
    settings = get_settings()
    if settings.session_isolation:
        # Public deployments: each browser session gets its own private, temporary store.
        return Path(tempfile.gettempdir()) / "documind-sessions" / st.session_state.session_id
    return Path(settings.chroma_persist_dir)


def _fp(value: str | None) -> str:
    return hashlib.sha256((value or "").encode()).hexdigest()[:12]


def gemini_key() -> str | None:
    return get_settings().gemini_api_key or st.session_state.get("user_api_key") or None


def shopaikey_key() -> str | None:
    return get_settings().shopaikey_api_key or st.session_state.get("shopaikey_key") or None


def current_selection() -> GenerationSelection:
    settings = get_settings()
    return GenerationSelection(
        provider=st.session_state.llm_provider,
        gemini_key=gemini_key(),
        gemini_model=settings.gemini_model,
        shopaikey_key=shopaikey_key(),
        shopaikey_protocol=st.session_state.shopaikey_protocol,
        shopaikey_model=(st.session_state.shopaikey_model or "").strip(),
        shopaikey_root=settings.shopaikey_base_url,
    )


def selection_ident(sel: GenerationSelection | None = None) -> str:
    """Identifies a provider configuration without containing any key material."""
    sel = sel or current_selection()
    if sel.provider == "gemini":
        return f"gemini|{sel.gemini_model}|{_fp(sel.gemini_key)}"
    if sel.provider == "shopaikey":
        return f"shopaikey|{sel.shopaikey_protocol}|{sel.shopaikey_model}|{_fp(sel.shopaikey_key)}"
    return "offline"


def get_app() -> DocuMindApp:
    base = _shared_app(str(persist_dir().resolve()))
    ident = selection_ident()
    cached = st.session_state.get("_generator")
    if not cached or cached[0] != ident:
        generator, notice = build_generator(get_settings(), current_selection())
        st.session_state["_generator"] = (ident, generator, notice)
    _, generator, notice = st.session_state["_generator"]
    return with_generator(base, generator, [notice] if notice else None)


# --------------------------------------------------------------------------- document scope
def sync_scope(app: DocuMindApp) -> list[str]:
    """Keep the selection valid and auto-select documents indexed since the last run."""
    searchable = app.ingestion.searchable_ids()
    scope = st.session_state.doc_scope
    if scope is None:
        scope = list(searchable)
    new = [d for d in searchable if d not in st.session_state.known_docs]
    scope = [d for d in scope if d in searchable] + [d for d in new if d not in scope]
    st.session_state.known_docs = list(searchable)
    st.session_state.doc_scope = scope
    return searchable


def display_name(filename: str, limit: int = 30) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem if len(stem) <= limit else stem[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- provider status
@dataclass
class ModelStatus:
    label: str  # compact, e.g. "ShopAIKey · gpt-5-mini"
    state: str  # key of STATE_TEXT
    kind: str  # dot colour: ok | accent | warn | err | muted
    detail: str


def active_model_label(app: DocuMindApp) -> str:
    sel = current_selection()
    if sel.provider == "offline":
        return "Offline retrieval"
    model = sel.gemini_model if sel.provider == "gemini" else sel.shopaikey_model
    return f"{PROVIDER_LABELS[sel.provider]} · {model or 'no model'}"


def model_status(app: DocuMindApp) -> ModelStatus:
    sel = current_selection()
    label = active_model_label(app)
    if sel.provider == "offline":
        return ModelStatus(label, "offline", "muted", OFFLINE_DETAIL)
    if app.generator is None:
        notice = st.session_state["_generator"][2] if st.session_state.get("_generator") else ""
        return ModelStatus(label, "not_configured", "warn", notice or "Provider is not configured.")
    record = st.session_state.conn.get(selection_ident(sel))
    if record:
        kind = "ok" if record["state"] == "connected" else "err"
        return ModelStatus(label, record["state"], kind, record["message"])
    return ModelStatus(label, "ready", "accent", "Configured but not verified. Use Test connection in Settings.")


def record_state(state: str, message: str) -> None:
    st.session_state.conn[selection_ident()] = {"state": state, "message": message, "at": time.time()}


def test_connection(app: DocuMindApp) -> None:
    """One real, minimal request to the selected provider; result is remembered per configuration."""
    if app.generator is None:
        record_state("not_configured", "Nothing to test: the provider is not configured.")
        return
    try:
        reported = app.generator.check()
    except GenerationError as exc:
        record_state(exc.state, str(exc))
        return
    record_state("connected", f"Verified {datetime.now(UTC):%H:%M} UTC with a real request (model: {reported}).")


def note_answer(answer: RAGAnswer) -> bool:
    """A real generated answer proves the connection; a provider error disproves it.

    Returns True when the displayed status changed (the sidebar needs a refresh).
    """
    ident = selection_ident()
    before = st.session_state.conn.get(ident, {}).get("state")
    if answer.mode == "generated":
        record_state("connected", "Last answer generated successfully.")
    elif answer.mode == "error" and answer.error_state:
        record_state(answer.error_state, answer.answer)
    else:
        return False
    return st.session_state.conn[ident]["state"] != before


def test_embeddings(app: DocuMindApp) -> tuple[bool, str]:
    check = getattr(app.embedder, "check", None)
    if check is None:
        return True, "The offline lexical index needs no connection."
    try:
        return True, f"Embedding endpoint answered ({check()})."
    except (GenerationError, EmbeddingError) as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- model discovery
def model_catalog() -> dict[str, Any] | None:
    key = (st.session_state.shopaikey_protocol, _fp(shopaikey_key()))
    return st.session_state.model_catalog.get(key)


def refresh_model_catalog() -> None:
    """Query the documented catalog once, on explicit request (never on rerun)."""
    protocol = st.session_state.shopaikey_protocol
    cache_key = (protocol, _fp(shopaikey_key()))
    settings = get_settings()
    try:
        models = list_shopaikey_models(
            protocol, api_key=shopaikey_key() or "", root=settings.shopaikey_base_url, timeout_s=20.0
        )
        entry = {"models": models, "error": None, "at": time.time()}
    except GenerationError as exc:
        entry = {"models": [], "error": str(exc), "state": exc.state, "at": time.time()}
    st.session_state.model_catalog[cache_key] = entry


def protocol_label(protocol: str) -> str:
    return PROTOCOLS.get(protocol, protocol)


# --------------------------------------------------------------------------- conversations
def current_conversation() -> dict[str, Any]:
    convs: dict[str, dict[str, Any]] = st.session_state.conversations
    cid = st.session_state.current_conv
    if cid not in convs:
        cid = uuid.uuid4().hex[:10]
        convs[cid] = {"title": "New chat", "messages": [], "created": datetime.now(UTC).isoformat()}
        st.session_state.current_conv = cid
    return convs[cid]


def new_conversation() -> None:
    convs: dict[str, dict[str, Any]] = st.session_state.conversations
    cid = st.session_state.current_conv
    if cid in convs and not convs[cid]["messages"]:
        del convs[cid]
    st.session_state.current_conv = None
    st.session_state.pop("pending_prompt", None)
    current_conversation()


def recent_conversations(limit: int = 8) -> list[tuple[str, str]]:
    convs: dict[str, dict[str, Any]] = st.session_state.conversations
    used = [(cid, c) for cid, c in convs.items() if c["messages"]]
    used.sort(key=lambda item: item[1]["created"], reverse=True)
    return [(cid, c["title"]) for cid, c in used[:limit]]


def history_for_model(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Earlier turns as plain text for the generator (only real generated answers)."""
    turns: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "user":
            turns.append({"role": "user", "content": message["content"]})
        else:
            answer = message.get("answer") or {}
            if answer.get("mode") == "generated" and not answer.get("insufficient_evidence"):
                turns.append({"role": "assistant", "content": answer.get("answer", "")})
            elif turns and turns[-1]["role"] == "user":
                turns.pop()  # drop a question that got no generated answer
    return turns
