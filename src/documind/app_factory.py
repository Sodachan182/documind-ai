"""Wire settings into concrete components (composition root).

Embeddings and answer generation are configured independently:

* the **embedder** (vector space) comes from server configuration only, because
  changing it requires re-indexing every document;
* the **generator** is a :class:`GenerationSelection` that the UI may change per
  session (provider, ShopAIKey protocol, model, session-only key).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from documind.config import Settings
from documind.embeddings.base import Embedder
from documind.embeddings.gemini_embeddings import GeminiEmbedder
from documind.embeddings.local_embeddings import HashingEmbedder
from documind.llm.base import AnswerGenerator
from documind.llm.errors import GenerationError
from documind.llm.gemini_client import GeminiAnswerGenerator
from documind.llm.shopaikey import base_url_for, gateway_root
from documind.llm.shopaikey import build_generator as build_shopaikey_generator
from documind.retrieval.retriever import Retriever
from documind.retrieval.vector_store import VectorStore
from documind.services.ingestion_service import IngestionService
from documind.services.rag_service import RAGService
from documind.services.registry import DocumentRegistry


@dataclass
class DocuMindApp:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    registry: DocumentRegistry
    ingestion: IngestionService
    retriever: Retriever
    rag: RAGService
    generator: AnswerGenerator | None
    notices: list[str] = field(default_factory=list)

    @property
    def generation_enabled(self) -> bool:
        return self.generator is not None


@dataclass(frozen=True)
class GenerationSelection:
    provider: str  # "gemini" | "shopaikey" | "offline"
    gemini_key: str | None = None
    gemini_model: str = ""
    shopaikey_key: str | None = None
    shopaikey_protocol: str = "openai"
    shopaikey_model: str = ""
    shopaikey_root: str = ""

    @classmethod
    def from_settings(cls, settings: Settings, *, gemini_key: str | None = None) -> GenerationSelection:
        key = settings.gemini_api_key if gemini_key is None else gemini_key
        provider = settings.llm_provider
        if provider == "auto":
            provider = "gemini" if key else ("shopaikey" if settings.has_shopaikey_key else "offline")
        return cls(
            provider=provider,
            gemini_key=key or None,
            gemini_model=settings.gemini_model,
            shopaikey_key=settings.shopaikey_api_key or None,
            shopaikey_protocol=settings.shopaikey_protocol,
            shopaikey_model=settings.shopaikey_model,
            shopaikey_root=settings.shopaikey_base_url,
        )


def build_generator(settings: Settings, selection: GenerationSelection) -> tuple[AnswerGenerator | None, str | None]:
    """Return ``(generator, notice)``; ``generator`` is None for offline or incomplete setups."""
    common = {
        "max_output_tokens": settings.max_output_tokens,
        "timeout_s": settings.request_timeout_s,
        "max_retries": settings.max_retries,
    }
    try:
        if selection.provider == "gemini":
            if not selection.gemini_key:
                return None, "Google Gemini is selected but no GEMINI_API_KEY is set."
            return GeminiAnswerGenerator(selection.gemini_key, selection.gemini_model, **common), None
        if selection.provider == "shopaikey":
            if not selection.shopaikey_key:
                return None, "ShopAIKey is selected but no ShopAIKey API key is set."
            if not selection.shopaikey_model:
                return None, "ShopAIKey is selected but no model is chosen."
            generator = build_shopaikey_generator(
                selection.shopaikey_protocol,
                api_key=selection.shopaikey_key,
                model=selection.shopaikey_model,
                root=selection.shopaikey_root,
                **common,
            )
            return generator, None
    except GenerationError as exc:
        return None, str(exc)
    return None, None


def build_embedder(settings: Settings, api_key: str | None) -> tuple[Embedder, list[str]]:
    """Embedder for the configured vector space. Fallbacks are reported, never silent."""
    notices: list[str] = []
    provider = settings.embedding_provider
    if provider == "auto":
        provider = "gemini" if api_key else "local"
    if provider == "gemini" and not api_key:
        notices.append(
            "DOCUMIND_EMBEDDING_PROVIDER=gemini but no GEMINI_API_KEY is set; using the offline lexical index."
        )
        provider = "local"
    if provider == "shopaikey":
        if not settings.has_shopaikey_key or not settings.embedding_model:
            notices.append(
                "DOCUMIND_EMBEDDING_PROVIDER=shopaikey needs SHOPAIKEY_API_KEY and DOCUMIND_EMBEDDING_MODEL; "
                "using the offline lexical index."
            )
            provider = "local"
        else:
            from documind.embeddings.shopaikey_embeddings import ShopAIKeyEmbedder

            try:
                url = base_url_for("openai", gateway_root(settings.shopaikey_base_url))
            except GenerationError as exc:
                notices.append(f"{exc} Using the offline lexical index.")
                return HashingEmbedder(), notices
            return (
                ShopAIKeyEmbedder(
                    settings.shopaikey_api_key or "",
                    settings.embedding_model,
                    settings.embedding_dimension,
                    base_url=url,
                    max_retries=settings.max_retries,
                    timeout_s=settings.request_timeout_s,
                ),
                notices,
            )
    if provider == "gemini":
        return (
            GeminiEmbedder(
                api_key=api_key or "",
                model=settings.embedding_model or settings.gemini_embedding_model,
                dimension=settings.embedding_dimension,
                max_retries=settings.max_retries,
                timeout_s=settings.request_timeout_s,
            ),
            notices,
        )
    return HashingEmbedder(), notices


def build_app(
    settings: Settings,
    *,
    api_key: str | None = None,
    persist_dir: Path | None = None,
    embedder: Embedder | None = None,
    generator: AnswerGenerator | None = None,
    selection: GenerationSelection | None = None,
) -> DocuMindApp:
    """``api_key`` overrides the Gemini key ("" = none); ``selection`` overrides generation."""
    key = api_key if api_key is not None else settings.gemini_api_key
    key = key.strip() if key else None
    notices: list[str] = []
    if embedder is None:
        embedder, notices = build_embedder(settings, key)
    if generator is None:
        selection = selection or GenerationSelection.from_settings(settings, gemini_key=key or "")
        generator, notice = build_generator(settings, selection)
        if notice:
            notices.append(notice)
    root = Path(persist_dir or settings.chroma_persist_dir)
    store = VectorStore(root, embedder.key, embedder.dimension)
    registry = DocumentRegistry(root)
    ingestion = IngestionService(
        embedder,
        store,
        registry,
        max_upload_bytes=settings.max_upload_bytes,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = Retriever(
        embedder,
        store,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        max_context_chars=settings.max_context_chars,
    )
    rag = RAGService(retriever, generator, max_question_chars=settings.max_question_chars)
    return DocuMindApp(settings, embedder, store, registry, ingestion, retriever, rag, generator, notices)


def with_generator(
    app: DocuMindApp, generator: AnswerGenerator | None, notices: list[str] | None = None
) -> DocuMindApp:
    """Same knowledge base and retriever, different answer generator (per session)."""
    rag = RAGService(app.retriever, generator, max_question_chars=app.settings.max_question_chars)
    return replace(app, generator=generator, rag=rag, notices=list(app.notices) + list(notices or []))
