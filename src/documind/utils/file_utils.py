"""Upload validation helpers: filename sanitisation, signatures, hashing."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath, PureWindowsPath

from documind.models import FileType

ALLOWED_EXTENSIONS: dict[str, FileType] = {".pdf": FileType.PDF, ".docx": FileType.DOCX}
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
MAX_FILENAME_CHARS = 150
# Guard against zip bombs disguised as DOCX files.
MAX_DOCX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024


class UploadValidationError(ValueError):
    """Raised for user-facing upload validation failures."""


def sanitize_filename(name: str) -> str:
    """Return a safe display/storage filename.

    Directory components are discarded (path traversal), reserved characters and
    Windows device names are neutralised, and length is capped while keeping the
    extension.
    """
    name = unicodedata.normalize("NFC", name or "")
    # Strip any directory part regardless of the separator style.
    name = PureWindowsPath(PurePosixPath(name.replace("\\", "/")).name).name
    name = _INVALID_CHARS.sub("_", name).strip(" .")
    name = re.sub(r"\s+", " ", name)
    if not name or set(name) <= {"_", "."}:
        name = "document"
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    ext = f".{ext.lower()}" if ext else ""
    max_stem = MAX_FILENAME_CHARS - len(ext)
    return f"{stem[:max_stem].rstrip(' .') or 'document'}{ext}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_file_type(filename: str, data: bytes) -> FileType:
    """Validate extension AND content signature; return the file type."""
    ext = PurePosixPath(filename.lower()).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"Unsupported file type '{ext or 'none'}'. Only PDF and DOCX files are accepted.")
    file_type = ALLOWED_EXTENSIONS[ext]
    if file_type == FileType.PDF:
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            raise UploadValidationError("The file has a .pdf extension but is not a valid PDF.")
    else:
        if not data.startswith(b"PK\x03\x04"):
            raise UploadValidationError("The file has a .docx extension but is not a valid DOCX.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
                total = sum(info.file_size for info in zf.infolist())
        except zipfile.BadZipFile as exc:
            raise UploadValidationError("The DOCX archive is corrupted.") from exc
        if "word/document.xml" not in names:
            raise UploadValidationError("The archive is not a Word document (word/document.xml missing).")
        if total > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise UploadValidationError("The DOCX file expands to an unsafe size and was rejected.")
    return file_type


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
