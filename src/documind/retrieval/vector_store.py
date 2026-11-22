"""ChromaDB persistence with explicit embeddings and model/dimension tracking.

Each embedding model + dimension gets its own collection, and the collection
metadata records which model produced its vectors. Opening a collection with a
different embedder raises :class:`IncompatibleCollectionError`, so vectors from
different spaces are never compared. Switching models therefore requires a
re-index (see ``IngestionService.reindex``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from documind.models import Chunk

COLLECTION_PREFIX = "documind_"


class IncompatibleCollectionError(RuntimeError):
    pass


def get_client(persist_dir: Path | str) -> chromadb.ClientAPI:
    """One consistently configured persistent client per directory."""
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
    )


def collection_name_for(embedding_key: str) -> str:
    return COLLECTION_PREFIX + hashlib.sha1(embedding_key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class VectorStore:
    def __init__(self, persist_dir: Path | str, embedding_key: str, dimension: int) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_key = embedding_key
        self.dimension = dimension
        self._client = get_client(self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name_for(embedding_key),
            metadata={
                "hnsw:space": "cosine",
                "embedding_key": embedding_key,
                "dimension": dimension,
            },
            embedding_function=None,  # we always pass explicit vectors
        )
        meta = self._collection.metadata or {}
        if meta.get("embedding_key") != embedding_key or int(meta.get("dimension", -1)) != dimension:
            raise IncompatibleCollectionError(
                f"Collection {self._collection.name} was built with "
                f"{meta.get('embedding_key')} ({meta.get('dimension')} dims), "
                f"not {embedding_key} ({dimension} dims)."
            )

    # -- writes -----------------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        for vec in embeddings:
            if len(vec) != self.dimension:
                raise IncompatibleCollectionError(f"Vector has {len(vec)} dims; collection expects {self.dimension}.")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
            embeddings=[list(map(float, v)) for v in embeddings],
        )

    def delete_document(self, document_id: str) -> None:
        """Delete a document's vectors from *every* DocuMind collection."""
        for col in self._client.list_collections():
            name = col if isinstance(col, str) else col.name
            if name.startswith(COLLECTION_PREFIX):
                self._client.get_collection(name).delete(where={"document_id": document_id})

    # -- reads ------------------------------------------------------------------
    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return self._collection.count()
        got = self._collection.get(where={"document_id": document_id}, include=[])
        return len(got["ids"])

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[str],
    ) -> list[tuple[Chunk, float]]:
        """Return ``(chunk, cosine_distance)`` pairs restricted to ``document_ids``.

        An empty ``document_ids`` list returns nothing (never "search everything").
        """
        if not document_ids or top_k <= 0:
            return []
        total = self._collection.count()
        if total == 0:
            return []
        where: dict[str, Any] = (
            {"document_id": document_ids[0]} if len(document_ids) == 1 else {"document_id": {"$in": list(document_ids)}}
        )
        result = self._collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        return [
            (Chunk.from_metadata(doc or "", dict(meta or {})), float(dist))
            for _id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False)
        ]
