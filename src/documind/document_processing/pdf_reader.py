"""PDF text extraction with PyMuPDF, preserving exact page numbers."""

from __future__ import annotations

import pymupdf

from documind.document_processing.cleaner import clean_paragraph
from documind.models import ExtractedDocument, FileType, TextBlock

# Below this average number of characters per page we assume a scanned PDF.
MIN_CHARS_PER_PAGE = 40


class DocumentExtractionError(Exception):
    """Raised when a document cannot be parsed."""


def extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several internal error types
        raise DocumentExtractionError("The PDF file is corrupted or cannot be opened.") from exc

    with doc:
        if doc.needs_pass:
            raise DocumentExtractionError("Password-protected PDFs are not supported.")
        blocks: list[TextBlock] = []
        empty_pages: list[int] = []
        for page_number, page in enumerate(doc, start=1):
            page_blocks = 0
            # "blocks" mode returns paragraph-like text blocks in reading order.
            for raw in page.get_text("blocks", sort=True):
                if raw[6] != 0:  # 1 = image block
                    continue
                text = clean_paragraph(raw[4])
                if text:
                    blocks.append(TextBlock(text=text, page_number=page_number))
                    page_blocks += 1
            if page_blocks == 0:
                empty_pages.append(page_number)
        page_count = doc.page_count

    total_chars = sum(len(b.text) for b in blocks)
    needs_ocr = page_count > 0 and total_chars / page_count < MIN_CHARS_PER_PAGE
    warnings: list[str] = []
    if needs_ocr:
        warnings.append(
            "Little or no extractable text was found. This looks like a scanned PDF; "
            "OCR is required before it can be searched."
        )
    elif empty_pages:
        shown = ", ".join(map(str, empty_pages[:10]))
        more = "..." if len(empty_pages) > 10 else ""
        warnings.append(f"No extractable text on page(s) {shown}{more} (images or scans?).")

    return ExtractedDocument(
        filename=filename,
        file_type=FileType.PDF,
        blocks=blocks,
        page_count=page_count,
        warnings=warnings,
        needs_ocr=needs_ocr,
    )
