"""Paragraph- and section-aware chunking that never crosses citation boundaries.

A chunk never spans two PDF pages or two DOCX sections, so every chunk maps to
exactly one page (PDF) or one section (DOCX) and citations stay accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from documind.models import Chunk, ExtractedDocument, TextBlock

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class _Piece:
    text: str
    block: TextBlock


def _split_long(text: str, limit: int) -> list[str]:
    """Split text longer than ``limit`` at sentence, then word boundaries."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        units = [sentence] if len(sentence) <= limit else sentence.split(" ")
        for unit in units:
            if len(unit) > limit:  # pathological token (e.g. a URL); hard split
                unit_parts = [unit[i : i + limit] for i in range(0, len(unit), limit)]
            else:
                unit_parts = [unit]
            for part in unit_parts:
                candidate = f"{current} {part}".strip() if current else part
                if len(candidate) <= limit:
                    current = candidate
                else:
                    if current:
                        parts.append(current)
                    current = part
    if current:
        parts.append(current)
    return parts


def _overlap_tail(text: str, overlap: int) -> str:
    """Last ~``overlap`` characters of ``text``, starting on a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return "" if overlap <= 0 else text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def _unit_key(block: TextBlock) -> tuple[object, ...]:
    return (block.page_number, block.section_heading)


def chunk_document(
    doc: ExtractedDocument,
    *,
    document_id: str,
    file_hash: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # 1. Group consecutive blocks by citation unit (page or section).
    units: list[list[TextBlock]] = []
    for block in doc.blocks:
        if units and _unit_key(units[-1][-1]) == _unit_key(block):
            units[-1].append(block)
        else:
            units.append([block])

    chunks: list[Chunk] = []
    piece_limit = chunk_size - chunk_overlap - 1  # room for overlap + newline

    def emit(text: str, first: TextBlock, last: TextBlock) -> None:
        index = len(chunks)
        para_start = first.paragraph_index
        para_end = last.paragraph_index
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:{index:04d}",
                document_id=document_id,
                filename=doc.filename,
                file_hash=file_hash,
                file_type=doc.file_type,
                content=text,
                chunk_index=index,
                page_number=first.page_number,
                section_heading=first.section_heading,
                paragraph_index=para_start,
                paragraph_end=para_end if para_end != para_start else None,
                table_index=first.table_index if first.kind == "table" else None,
            )
        )

    # 2. Pack paragraph pieces greedily inside each unit, with overlap.
    for unit in units:
        pieces = [_Piece(part, block) for block in unit for part in _split_long(block.text, piece_limit)]
        current_text = ""
        current_first: TextBlock | None = None
        current_last: TextBlock | None = None
        has_new_content = False
        for piece in pieces:
            # Tables are kept separate from prose so their location label is exact.
            kind_change = current_last is not None and (
                piece.block.kind != current_last.kind
                or (piece.block.kind == "table" and piece.block is not current_last)
            )
            candidate = f"{current_text}\n{piece.text}" if current_text else piece.text
            if current_first is not None and (len(candidate) > chunk_size or kind_change):
                if has_new_content:
                    emit(current_text, current_first, current_last)  # type: ignore[arg-type]
                tail = "" if kind_change else _overlap_tail(current_text, chunk_overlap)
                current_text = f"{tail}\n{piece.text}" if tail else piece.text
                current_first = piece.block
                current_last = piece.block
                has_new_content = True
                continue
            current_text = candidate
            current_first = current_first or piece.block
            current_last = piece.block
            has_new_content = True
        if current_first is not None and has_new_content:
            emit(current_text, current_first, current_last)  # type: ignore[arg-type]
    return chunks
