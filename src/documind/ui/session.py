"""Session state, app construction and model status for the Streamlit UI.

Durable state lives under plain keys in ``st.session_state``; widgets use their
own keys (Streamlit discards widget state on pages where the widget is not
rendered) and are synced from/to the durable keys.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from documind.app_factory import DocuMindApp, build_app
from documind.config import Settings, load_settings
from documind.embeddings.base import EmbeddingError
from documind.embeddings.gemini_embeddings import GeminiEmbedder
from documind.llm.errors import GenerationError
from documind.models import RAGAnswer

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "demo" / "documents"
QUESTIONS_FILE = ROOT / "demo" / "questions.json"
RESULTS_DIR = ROOT / "docs" / "evaluation_results"
_SECRET_KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL", "DOCUMIND_SESSION_ISOLATION")


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
    """App using the server-side key (if any). Shared by sessions using the same store."""
    return build_app(get_settings(), persist_dir=Path(persist_dir))


def init_state() -> None:
    settings = get_settings()
    defaults: dict[str, Any] = {
        "session_id": uuid.uuid4().hex,
        "conversations": {},
        "current_conv": None,
        "user_api_key": "",
        "doc_scope": None,
        "known_docs": [],
        "uploader_key": 0,
        "top_k": settings.top_k,
        "gemini_check": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_dir() -> Path:
    settings = get_settings()
    if settings.session_isolation:
        # Public deployments: each browser session gets its own private, temporary store.
        return Path(tempfile.gettempdir()) / "documind-sessions" / st.session_state.session_id
    return Path(settings.chroma_persist_dir)


def active_key() -> str | None:
    return get_settings().gemini_api_key or st.session_state.get("user_api_key") or None


def key_fingerprint() -> str:
    return hashlib.sha256((active_key() or "").encode()).hexdigest()[:12]


def get_app() -> DocuMindApp:
    settings = get_settings()
    pdir = str(persist_dir().resolve())
    user_key = st.session_state.get("user_api_key")
    if user_key and not settings.gemini_api_key:
        # A key typed into the UI is held only in this session's state (server memory),
        # never in the process-wide cache, so it is discarded with the session.
        ident = (key_fingerprint(), pdir)
        cached = st.session_state.get("_user_app")
        if not cached or cached[0] != ident:
            st.session_state["_user_app"] = (ident, build_app(settings, api_key=user_key, persist_dir=Path(pdir)))
        return st.session_state["_user_app"][1]
    st.session_state.pop("_user_app", None)
    return _shared_app(pdir)


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


# --------------------------------------------------------------------------- model status
@dataclass
class ModelStatus:
    label: str
    kind: str  # ok | accent | err | muted
    detail: str


def model_status(app: DocuMindApp) -> ModelStatus:
    if not app.generation_enabled:
        return ModelStatus("Offline retrieval", "muted", "No Gemini API key. Answers show retrieved passages only.")
    check = st.session_state.get("gemini_check")
    if check and check["fp"] == key_fingerprint():
        if check["ok"]:
            return ModelStatus("Gemini connected", "ok", check["message"])
        return ModelStatus("Gemini error", "err", check["message"])
    return ModelStatus("Gemini configured", "accent", "Key present but not verified yet. Use Test connection.")


def record_check(ok: bool, message: str) -> None:
    st.session_state.gemini_check = {"fp": key_fingerprint(), "ok": ok, "message": message}


def test_connection(app: DocuMindApp) -> None:
    """Verify key + models via the free ``models.get`` endpoint and remember the result."""
    if app.generator is None:
        record_check(False, "No API key configured.")
        return
    try:
        app.generator.check()  # type: ignore[attr-defined]
        if isinstance(app.embedder, GeminiEmbedder):
            app.embedder.check()
    except (GenerationError, EmbeddingError) as exc:
        record_check(False, str(exc))
        return
    record_check(True, f"Verified {datetime.now(UTC):%H:%M} UTC: key accepted, models available.")


def note_answer(answer: RAGAnswer) -> bool:
    """A real generated answer proves the connection; an API error disproves it.

    Returns True when the displayed status changed (the sidebar needs a refresh).
    """
    before = st.session_state.get("gemini_check")
    if answer.mode == "generated":
        record_check(True, "Last answer generated successfully.")
    elif answer.mode == "error" and "Gemini" in answer.answer:
        record_check(False, answer.answer)
    else:
        return False
    after = st.session_state.gemini_check
    return not before or before["ok"] != after["ok"] or before["fp"] != after["fp"]


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
    # Drop the current conversation if it was never used.
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
