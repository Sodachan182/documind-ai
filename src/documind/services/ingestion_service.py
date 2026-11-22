"""Ingestion orchestration: validate -> extract -> chunk -> embed -> index."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from documind.document_processing.chunker import chunk_document
from documind.document_processing.pdf_reader import DocumentExtractionError
from documind.embeddings.base import Embedder, EmbeddingError
from documind.models import DocumentRecord, DocumentStatus
from documind.retrieval.vector_store import VectorStore
from documind.services.document_service import extract_document, validate_upload
from documind.services.registry import DocumentRegistry
from documind.utils.file_utils import UploadValidationError
from documind.utils.logging_utils import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]
IngestOutcome = Literal["indexed", "duplicate", "needs_ocr", "rejected", "failed"]


class IngestionResult(BaseModel):
    outcome: IngestOutcome
    filename: str
    message: str
    record: DocumentRecord | None = None


def _noop(_: float, __: str) -> None:
    return None


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        registry: DocumentRegistry,
        *,
        max_upload_bytes: int,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        embed_batch_size: int = 32,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.registry = registry
        self.max_upload_bytes = max_upload_bytes
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_batch_size = embed_batch_size

    # ------------------------------------------------------------------ ingest
    def ingest(self, filename: str, data: bytes, progress: ProgressCallback | None = None) -> IngestionResult:
        report = progress or _noop
        report(0.05, "Validating file")
        try:
            upload = validate_upload(filename, data, self.max_upload_bytes)
        except UploadValidationError as exc:
            return IngestionResult(outcome="rejected", filename=filename, message=str(exc))

        existing = self.registry.find_by_hash(upload.file_hash)
        if existing and existing.status != DocumentStatus.FAILED:
            return IngestionResult(
                outcome="duplicate",
                filename=upload.filename,
                message=f"Identical content is already in the knowledge base as '{existing.filename}'.",
                record=existing,
            )

        report(0.15, "Extracting text")
        base = DocumentRecord(
            document_id=upload.document_id,
            filename=upload.filename,
            file_hash=upload.file_hash,
            file_type=upload.file_type,
            size_bytes=upload.size_bytes,
            status=DocumentStatus.FAILED,
        )
        try:
            extracted = extract_document(upload, data)
        except DocumentExtractionError as exc:
            record = base.model_copy(update={"error": str(exc)})
            self.registry.upsert(record)
            return IngestionResult(outcome="failed", filename=upload.filename, message=str(exc), record=record)

        base = base.model_copy(
            update={
                "page_count": extracted.page_count,
                "paragraph_count": extracted.paragraph_count,
                "char_count": extracted.char_count,
                "warnings": extracted.warnings,
            }
        )
        if extracted.needs_ocr or not extracted.blocks:
            status = DocumentStatus.NEEDS_OCR if extracted.needs_ocr else DocumentStatus.FAILED
            message = extracted.warnings[0] if extracted.warnings else "No text could be extracted."
            record = base.model_copy(update={"status": status, "error": message})
            self.registry.upsert(record)
            return IngestionResult(
                outcome="needs_ocr" if extracted.needs_ocr else "failed",
                filename=upload.filename,
                message=message,
                record=record,
            )

        report(0.3, "Chunking text")
        chunks = chunk_document(
            extracted,
            document_id=upload.document_id,
            file_hash=upload.file_hash,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        try:
            self._embed_and_store(chunks, report)
        except EmbeddingError as exc:
            self.store.delete_document(upload.document_id)  # no partial indexes
            record = base.model_copy(update={"error": str(exc)})
            self.registry.upsert(record)
            return IngestionResult(outcome="failed", filename=upload.filename, message=str(exc), record=record)

        self.registry.save_chunks(upload.document_id, chunks)
        record = base.model_copy(
            update={
                "status": DocumentStatus.INDEXED,
                "chunk_count": len(chunks),
                "embedding_key": self.embedder.key,
                "error": None,
            }
        )
        self.registry.upsert(record)
        report(1.0, "Indexed")
        logger.info("Indexed %s (%d chunks)", record.document_id, len(chunks))
        return IngestionResult(
            outcome="indexed",
            filename=upload.filename,
            message=f"Indexed {len(chunks)} chunks.",
            record=record,
        )

    def _embed_and_store(self, chunks: list, report: ProgressCallback) -> None:
        total = len(chunks)
        for start in range(0, total, self.embed_batch_size):
            batch = chunks[start : start + self.embed_batch_size]
            vectors = self.embedder.embed_documents([c.embedding_text() for c in batch])
            self.store.upsert(batch, vectors)
            done = min(total, start + len(batch))
            report(0.3 + 0.65 * done / max(total, 1), f"Embedding chunks {done}/{total}")

    # ------------------------------------------------------------- management
    def list_documents(self) -> list[DocumentRecord]:
        return self.registry.all()

    def searchable_ids(self) -> list[str]:
        """Documents indexed in the *current* embedding space."""
        return [
            r.document_id
            for r in self.registry.all()
            if r.status == DocumentStatus.INDEXED and r.embedding_key == self.embedder.key
        ]

    def needs_reindex(self, record: DocumentRecord) -> bool:
        return record.status == DocumentStatus.INDEXED and record.embedding_key != self.embedder.key

    def delete(self, document_id: str) -> bool:
        if self.registry.get(document_id) is None:
            return False
        self.store.delete_document(document_id)
        self.registry.remove(document_id)
        return True

    def reindex(self, document_id: str, progress: ProgressCallback | None = None) -> DocumentRecord:
        record = self.registry.get(document_id)
        if record is None:
            raise KeyError(document_id)
        chunks = self.registry.load_chunks(document_id)
        if not chunks:
            raise EmbeddingError("No stored chunks for this document; please upload it again.")
        self._embed_and_store(chunks, progress or _noop)
        updated = record.model_copy(update={"embedding_key": self.embedder.key})
        self.registry.upsert(updated)
        return updated
