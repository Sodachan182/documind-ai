"""Application configuration loaded from environment variables / `.env`.

Secrets are never logged: the API key field is excluded from ``repr``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Defaults verified against https://ai.google.dev/gemini-api/docs/models (Nov 2026).
# ``gemini-flash-latest`` is Google's alias that always points at the current
# stable Flash release, so the default does not silently go stale. Pin a
# specific model (e.g. ``gemini-3.8-flash``) via GEMINI_MODEL for reproducibility.
DEFAULT_GENERATION_MODEL = "gemini-flash-latest"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSION = 768

EmbeddingProvider = Literal["auto", "gemini", "local"]


class Settings(BaseModel):
    """Runtime settings. Construct with :func:`load_settings`."""

    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_model: str = DEFAULT_GENERATION_MODEL
    gemini_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, ge=64, le=3072)
    embedding_provider: EmbeddingProvider = "auto"

    chroma_persist_dir: Path = Path("./data/chroma")
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=12000, ge=1000)
    max_question_chars: int = Field(default=2000, ge=50)

    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=8)
    session_isolation: bool = False

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolved_embedding_provider(self) -> Literal["gemini", "local"]:
        """``auto`` uses Gemini when a key is configured, otherwise the offline embedder."""
        if self.embedding_provider == "auto":
            return "gemini" if self.has_api_key else "local"
        return self.embedding_provider


_ENV_MAP: dict[str, str] = {
    "GEMINI_API_KEY": "gemini_api_key",
    "GEMINI_MODEL": "gemini_model",
    "GEMINI_EMBEDDING_MODEL": "gemini_embedding_model",
    "GEMINI_EMBEDDING_DIMENSION": "embedding_dimension",
    "EMBEDDING_PROVIDER": "embedding_provider",
    "CHROMA_PERSIST_DIR": "chroma_persist_dir",
    "MAX_UPLOAD_MB": "max_upload_mb",
    "RAG_TOP_K": "top_k",
    "RAG_CHUNK_SIZE": "chunk_size",
    "RAG_CHUNK_OVERLAP": "chunk_overlap",
    "RAG_MIN_SIMILARITY": "min_similarity",
    "RAG_MAX_CONTEXT_CHARS": "max_context_chars",
    "GEMINI_TIMEOUT_S": "request_timeout_s",
    "GEMINI_MAX_RETRIES": "max_retries",
    "DOCUMIND_SESSION_ISOLATION": "session_isolation",
}


def load_settings(
    env_file: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    use_dotenv: bool = True,
) -> Settings:
    """Build :class:`Settings` from the process environment (and optionally `.env`)."""
    if use_dotenv:
        load_dotenv(env_file, override=False)
    values: dict[str, Any] = {}
    for env_name, field_name in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip() != "":
            values[field_name] = raw.strip()
    # GOOGLE_API_KEY is the SDK's alternative variable name.
    if "gemini_api_key" not in values and os.environ.get("GOOGLE_API_KEY", "").strip():
        values["gemini_api_key"] = os.environ["GOOGLE_API_KEY"].strip()
    if overrides:
        values.update(overrides)
    return Settings(**values)
