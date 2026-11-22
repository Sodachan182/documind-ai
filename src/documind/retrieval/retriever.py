"""Query embedding + semantic search + context filtering."""

from __future__ import annotations

from collections.abc import Sequence

from documind.embeddings.base import Embedder
from documind.models import RetrievedChunk
from documind.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        max_context_chars: int = 12000,
    ) -> None:
        if embedder.key != store.embedding_key:
            raise ValueError("Embedder and vector store use different embedding spaces.")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.max_context_chars = max_context_chars

    def retrieve(self, query: str, document_ids: Sequence[str], top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed the query, search only ``document_ids`` and filter the context.

        Filtering: drop chunks from unselected documents (defence in depth), below
        ``min_similarity``, or duplicating earlier content, and stop once the
        context character budget is exhausted.
        """
        if not document_ids or not query.strip():
            return []
        k = top_k or self.top_k
        vector = self.embedder.embed_query(query)
        raw = self.store.query(vector, top_k=k, document_ids=document_ids)

        allowed = set(document_ids)
        seen_content: set[str] = set()
        results: list[RetrievedChunk] = []
        budget = self.max_context_chars
        for chunk, distance in raw:
            if chunk.document_id not in allowed:
                continue
            similarity = 1.0 - distance  # cosine similarity in [-1, 1]; not a probability
            if similarity < self.min_similarity:
                continue
            fingerprint = " ".join(chunk.content.split()).lower()
            if fingerprint in seen_content:
                continue
            if len(chunk.content) > budget and results:
                break
            seen_content.add(fingerprint)
            budget -= len(chunk.content)
            results.append(RetrievedChunk(chunk=chunk, similarity=round(similarity, 4), rank=len(results) + 1))
        return results
