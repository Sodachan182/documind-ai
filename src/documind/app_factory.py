"""Wire settings into concrete components (composition root)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from documind.config import Settings
from documind.embeddings.base import Embedder
from documind.embeddings.gemini_embeddings import GeminiEmbedder
from documind.embeddings.local_embeddings import HashingEmbedder
from documind.llm.gemini_client import AnswerGenerator, GeminiAnswerGenerator
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
    notices: list[str]

    @property
    def generation_enabled(self) -> bool:
        return self.generator is not None


def build_embedder(settings: Settings, api_key: str | None) -> tuple[Embedder, list[str]]:
    notices: list[str] = []
    provider = settings.embedding_provider
    if provider == "auto":
        provider = "gemini" if api_key else "local"
    if provider == "gemini" and not api_key:
        notices.append("EMBEDDING_PROVIDER=gemini but no API key is set; using offline lexical embeddings.")
        provider = "local"
    if provider == "gemini":
        return (
            GeminiEmbedder(
                api_key=api_key or "",
                model=settings.gemini_embedding_model,
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
) -> DocuMindApp:
    key = api_key if api_key is not None else settings.gemini_api_key
    key = key.strip() if key else None
    notices: list[str] = []
    if embedder is None:
        embedder, notices = build_embedder(settings, key)
    if generator is None and key:
        generator = GeminiAnswerGenerator(
            api_key=key,
            model=settings.gemini_model,
            max_retries=settings.max_retries,
            timeout_s=settings.request_timeout_s,
        )
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
