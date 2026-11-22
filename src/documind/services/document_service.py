"""Upload validation and text extraction (in memory, never written to disk)."""

from __future__ import annotations

from pydantic import BaseModel

from documind.document_processing.docx_reader import extract_docx
from documind.document_processing.pdf_reader import extract_pdf
from documind.models import ExtractedDocument, FileType
from documind.utils.file_utils import (
    UploadValidationError,
    detect_file_type,
    human_size,
    sanitize_filename,
    sha256_bytes,
)


class ValidatedUpload(BaseModel):
    filename: str
    file_type: FileType
    file_hash: str
    size_bytes: int

    @property
    def document_id(self) -> str:
        # Content-derived ID: identical files always map to the same document.
        return f"doc_{self.file_hash[:16]}"


def validate_upload(filename: str, data: bytes, max_bytes: int) -> ValidatedUpload:
    if not data:
        raise UploadValidationError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadValidationError(f"The file is {human_size(len(data))}; the limit is {human_size(max_bytes)}.")
    safe_name = sanitize_filename(filename)
    file_type = detect_file_type(safe_name, data)
    return ValidatedUpload(
        filename=safe_name,
        file_type=file_type,
        file_hash=sha256_bytes(data),
        size_bytes=len(data),
    )


def extract_document(upload: ValidatedUpload, data: bytes) -> ExtractedDocument:
    if upload.file_type == FileType.PDF:
        return extract_pdf(data, upload.filename)
    return extract_docx(data, upload.filename)
