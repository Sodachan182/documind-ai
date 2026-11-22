"""Application configuration loaded from environment variables / `.env`.

Secrets are never logged: API key fields are excluded from ``repr``.

Generation and embeddings are configured independently:

* ``DOCUMIND_LLM_PROVIDER`` picks who writes answers (Gemini direct, the
  ShopAIKey gateway, or nobody = retrieval only).
* ``DOCUMIND_EMBEDDING_PROVIDER`` picks the vector space used for search
  (Gemini, ShopAIKey's OpenAI-format embeddings, or the offline lexical index).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Defaults verified against https://ai.google.dev/gemini-api/docs/models (Nov 2026).
# ``gemini-flash-latest`` is Google's alias that always points at the current
# stable Flash release, so the default does not silently go stale. Pin a
# specific model (e.g. ``gemini-3.8-flash``) via GEMINI_MODEL for reproducibility.
DEFAULT_GENERATION_MODEL = "gemini-flash-latest"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSION = 768
# Documented in https://shopaikey.com/docs/quickstart. The OpenAI-format SDK base URL
# adds /v1; the Anthropic- and Gemini-format SDKs use this root unchanged.
DEFAULT_SHOPAIKEY_ROOT = "https://api.shopaikey.com"

# "local" is the internal name of the offline lexical index; it is part of existing
# collection keys, so it is kept. "offline" is accepted as a friendlier alias.
EmbeddingProvider = Literal["auto", "gemini", "shopaikey", "local"]
LLMProvider = Literal["auto", "gemini", "shopaikey", "offline"]
ShopAIKeyProtocol = Literal["openai", "anthropic", "gemini"]


class Settings(BaseModel):
    """Runtime settings. Construct with :func:`load_settings`."""

    # --- answer generation
    llm_provider: LLMProvider = "auto"
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_model: str = DEFAULT_GENERATION_MODEL
    shopaikey_api_key: str | None = Field(default=None, repr=False)
    shopaikey_base_url: str = DEFAULT_SHOPAIKEY_ROOT
    shopaikey_model: str = ""
    shopaikey_protocol: ShopAIKeyProtocol = "openai"
    max_output_tokens: int = Field(default=1024, ge=64, le=8192)

    # --- embeddings (independent of generation)
    embedding_provider: EmbeddingProvider = "auto"
    embedding_model: str = ""  # empty = provider default (Gemini: gemini_embedding_model)
    gemini_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, ge=64, le=4096)

    chroma_persist_dir: Path = Path("./data/chroma")
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=12000, ge=1000)
    max_question_chars: int = Field(default=2000, ge=50)

    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    session_isolation: bool = False

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def _offline_alias(cls, value: Any) -> Any:
        return "local" if isinstance(value, str) and value.strip().lower() == "offline" else value

    @field_validator("llm_provider", "shopaikey_protocol", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self

    @property
    def has_api_key(self) -> bool:
        """A direct Google Gemini key is configured on the server."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def has_shopaikey_key(self) -> bool:
        return bool(self.shopaikey_api_key and self.shopaikey_api_key.strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolved_llm_provider(self) -> Literal["gemini", "shopaikey", "offline"]:
        """``auto``: Gemini if its key is set, else ShopAIKey if its key is set, else offline."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.has_api_key:
            return "gemini"
        if self.has_shopaikey_key:
            return "shopaikey"
        return "offline"

    def resolved_embedding_provider(self) -> Literal["gemini", "shopaikey", "local"]:
        """``auto`` uses Gemini when its key is configured, otherwise the offline index.

        ShopAIKey embeddings are never chosen automatically: they must be enabled
        explicitly together with a model and its dimension.
        """
        if self.embedding_provider == "auto":
            return "gemini" if self.has_api_key else "local"
        return self.embedding_provider


# field -> environment variable names, preferred name first (later ones are legacy aliases).
_ENV_MAP: dict[str, tuple[str, ...]] = {
    "llm_provider": ("DOCUMIND_LLM_PROVIDER",),
    "gemini_api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini_model": ("GEMINI_MODEL",),
    "shopaikey_api_key": ("SHOPAIKEY_API_KEY",),
    "shopaikey_base_url": ("SHOPAIKEY_BASE_URL",),
    "shopaikey_model": ("SHOPAIKEY_MODEL",),
    "shopaikey_protocol": ("SHOPAIKEY_PROTOCOL",),
    "max_output_tokens": ("DOCUMIND_MAX_OUTPUT_TOKENS",),
    "embedding_provider": ("DOCUMIND_EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER"),
    "embedding_model": ("DOCUMIND_EMBEDDING_MODEL",),
    "gemini_embedding_model": ("GEMINI_EMBEDDING_MODEL",),
    "embedding_dimension": ("DOCUMIND_EMBEDDING_DIMENSION", "GEMINI_EMBEDDING_DIMENSION"),
    "chroma_persist_dir": ("CHROMA_PERSIST_DIR",),
    "max_upload_mb": ("MAX_UPLOAD_MB",),
    "top_k": ("RAG_TOP_K",),
    "chunk_size": ("RAG_CHUNK_SIZE",),
    "chunk_overlap": ("RAG_CHUNK_OVERLAP",),
    "min_similarity": ("RAG_MIN_SIMILARITY",),
    "max_context_chars": ("RAG_MAX_CONTEXT_CHARS",),
    "request_timeout_s": ("DOCUMIND_REQUEST_TIMEOUT_S", "GEMINI_TIMEOUT_S"),
    "max_retries": ("DOCUMIND_MAX_RETRIES", "GEMINI_MAX_RETRIES"),
    "session_isolation": ("DOCUMIND_SESSION_ISOLATION",),
}

SECRET_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SHOPAIKEY_API_KEY")


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
    for field_name, env_names in _ENV_MAP.items():
        for env_name in env_names:
            raw = os.environ.get(env_name)
            if raw is not None and raw.strip() != "":
                values[field_name] = raw.strip()
                break
    if overrides:
        values.update(overrides)
    return Settings(**values)
