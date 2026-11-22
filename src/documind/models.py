"""Domain models shared across the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


class TextBlock(BaseModel):
    """A paragraph-level unit of extracted text with its source location."""

    text: str
    kind: Literal["paragraph", "table"] = "paragraph"
    page_number: int | None = None  # PDF only (1-based)
    section_heading: str | None = None  # DOCX only
    paragraph_index: int | None = None  # DOCX only (1-based, body order)
    table_index: int | None = None  # DOCX tables (1-based)


class ExtractedDocument(BaseModel):
    filename: str
    file_type: FileType
    blocks: list[TextBlock]
    page_count: int | None = None
    paragraph_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
    needs_ocr: bool = False

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    file_hash: str
    file_type: FileType
    content: str
    chunk_index: int
    page_number: int | None = None
    section_heading: str | None = None
    paragraph_index: int | None = None
    paragraph_end: int | None = None
    table_index: int | None = None

    @property
    def location_label(self) -> str:
        """Human-readable location. DOCX never gets an invented page number."""
        if self.file_type == FileType.PDF:
            return f"Page {self.page_number}" if self.page_number else "Page unknown"
        parts: list[str] = []
        if self.section_heading:
            parts.append(f"Section: {self.section_heading}")
        if self.table_index is not None:
            parts.append(f"Table {self.table_index}")
        elif self.paragraph_index is not None:
            if self.paragraph_end and self.paragraph_end != self.paragraph_index:
                parts.append(f"Paragraphs {self.paragraph_index}-{self.paragraph_end}")
            else:
                parts.append(f"Paragraph {self.paragraph_index}")
        return " · ".join(parts) or "Body"

    def embedding_text(self) -> str:
        """Text sent to the embedder: a small contextual header + the passage."""
        return f"Document: {self.filename}\nLocation: {self.location_label}\n\n{self.content}"

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata for ChromaDB (``None`` values are omitted)."""
        data = self.model_dump(exclude={"content"}, mode="json")
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_metadata(cls, content: str, metadata: dict[str, Any]) -> Chunk:
        return cls(content=content, **metadata)


class DocumentStatus(StrEnum):
    INDEXED = "indexed"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    file_type: FileType
    size_bytes: int
    page_count: int | None = None
    paragraph_count: int | None = None
    chunk_count: int = 0
    char_count: int = 0
    status: DocumentStatus
    embedding_key: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity: float  # cosine similarity (1 - cosine distance); NOT a probability
    rank: int


class Citation(BaseModel):
    """A retrieved passage exposed to the model under an app-controlled ID."""

    source_id: str  # "S1", "S2", ... assigned by the application
    chunk: Chunk
    similarity: float


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


AnswerMode = Literal["generated", "retrieval_only", "no_documents", "no_results", "error"]


class RAGAnswer(BaseModel):
    question: str
    answer: str
    language: Literal["en", "vi"]
    mode: AnswerMode
    sources: list[Citation] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    warnings: list[str] = Field(default_factory=list)
    usage: TokenUsage | None = None
    model: str | None = None
    provider: str | None = None  # human label of the generator, e.g. "ShopAIKey (OpenAI-compatible)"
    error_state: str | None = None  # e.g. auth_failed, model_unavailable (mode == "error")
    latency_ms: int | None = None

    @property
    def cited_sources(self) -> list[Citation]:
        by_id = {s.source_id: s for s in self.sources}
        return [by_id[i] for i in self.cited_ids if i in by_id]
