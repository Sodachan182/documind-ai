"""DOCX extraction (paragraphs + tables in body order) with section tracking."""

from __future__ import annotations

import io

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from documind.document_processing.cleaner import clean_paragraph
from documind.document_processing.pdf_reader import DocumentExtractionError
from documind.models import ExtractedDocument, FileType, TextBlock


def _is_heading(paragraph: Paragraph) -> bool:
    style = paragraph.style.name if paragraph.style is not None else ""
    return style.startswith("Heading") or style == "Title"


def _table_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = clean_paragraph(cell.text)
            if not cells or cells[-1] != text:  # merged cells repeat their text
                cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_docx(data: bytes, filename: str) -> ExtractedDocument:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentExtractionError("The DOCX file is corrupted or cannot be opened.") from exc

    blocks: list[TextBlock] = []
    section: str | None = None
    paragraph_index = 0
    table_index = 0
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = clean_paragraph(item.text)
            if not text:
                continue
            paragraph_index += 1
            if _is_heading(item):
                section = text
                continue  # headings are carried as section metadata, not as passages
            blocks.append(TextBlock(text=text, section_heading=section, paragraph_index=paragraph_index))
        elif isinstance(item, Table):
            text = _table_text(item)
            if not text:
                continue
            table_index += 1
            blocks.append(
                TextBlock(
                    text=text,
                    kind="table",
                    section_heading=section,
                    table_index=table_index,
                )
            )

    warnings: list[str] = []
    if not blocks:
        warnings.append("The DOCX file contains no extractable text.")
    return ExtractedDocument(
        filename=filename,
        file_type=FileType.DOCX,
        blocks=blocks,
        paragraph_count=paragraph_index,
        warnings=warnings,
    )
