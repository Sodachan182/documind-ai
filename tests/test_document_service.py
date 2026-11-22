"""Upload validation, sanitisation, duplicates, deletion, config."""

import pymupdf
import pytest

from documind.app_factory import DocuMindApp, build_app
from documind.config import Settings, load_settings
from documind.models import DocumentStatus, FileType
from documind.services.document_service import validate_upload
from documind.utils.file_utils import UploadValidationError, sanitize_filename
from documind.utils.logging_utils import redact


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\Windows\\system32\\evil.docx", "evil.docx"),
        ("C:\\Users\\x\\report.PDF", "report.pdf"),
        ('bad<>:"|?*name.pdf', "bad_______name.pdf"),
        ("CON.pdf", "_CON.pdf"),
        ("", "document"),
        ("Báo cáo tài chính 2026.docx", "Báo cáo tài chính 2026.docx"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_long_filename_is_truncated_keeping_extension() -> None:
    name = sanitize_filename("a" * 400 + ".pdf")
    assert len(name) <= 150 and name.endswith(".pdf")


def test_valid_uploads(sample_pdf: bytes, sample_docx: bytes) -> None:
    up = validate_upload("x.pdf", sample_pdf, 10_000_000)
    assert up.file_type == FileType.PDF and len(up.file_hash) == 64
    assert up.document_id == f"doc_{up.file_hash[:16]}"
    assert validate_upload("y.docx", sample_docx, 10_000_000).file_type == FileType.DOCX


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("notes.txt", b"hello", "Unsupported file type"),
        ("malware.exe", b"MZ....", "Unsupported file type"),
        ("fake.pdf", b"MZ not a pdf", "not a valid PDF"),
        ("fake.docx", b"%PDF-1.4", "not a valid DOCX"),
        ("empty.pdf", b"", "empty"),
    ],
)
def test_rejected_uploads(name: str, data: bytes, message: str) -> None:
    with pytest.raises(UploadValidationError, match=message):
        validate_upload(name, data, 10_000_000)


def test_zip_that_is_not_word_rejected() -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(UploadValidationError, match="not a Word document"):
        validate_upload("x.docx", buf.getvalue(), 10_000_000)


def test_size_limit(sample_pdf: bytes) -> None:
    with pytest.raises(UploadValidationError, match="limit"):
        validate_upload("x.pdf", sample_pdf, 100)


def test_ingest_and_duplicate_detection(app: DocuMindApp, sample_pdf: bytes) -> None:
    progress: list[float] = []
    first = app.ingestion.ingest("policy.pdf", sample_pdf, lambda p, _m: progress.append(p))
    assert first.outcome == "indexed"
    assert progress[-1] == 1.0 and progress == sorted(progress)
    record = first.record
    assert record.status == DocumentStatus.INDEXED and record.page_count == 2 and record.chunk_count >= 2
    dup = app.ingestion.ingest("renamed copy.pdf", sample_pdf)
    assert dup.outcome == "duplicate"
    assert len(app.ingestion.list_documents()) == 1


def test_rejected_upload_not_registered(app: DocuMindApp) -> None:
    result = app.ingestion.ingest("x.txt", b"hello")
    assert result.outcome == "rejected"
    assert app.ingestion.list_documents() == []


def test_scanned_pdf_recorded_as_needs_ocr(app: DocuMindApp) -> None:
    doc = pymupdf.open()
    doc.new_page()
    result = app.ingestion.ingest("scan.pdf", doc.tobytes())
    assert result.outcome == "needs_ocr"
    assert result.record.status == DocumentStatus.NEEDS_OCR
    assert app.store.count(result.record.document_id) == 0
    assert result.record.document_id not in app.ingestion.searchable_ids()


def test_embedding_failure_leaves_no_partial_index(settings: Settings, sample_pdf: bytes) -> None:
    from documind.embeddings.base import EmbeddingError
    from documind.embeddings.local_embeddings import HashingEmbedder

    class Flaky(HashingEmbedder):
        def embed_documents(self, texts):
            raise EmbeddingError("Gemini embedding failed: rate limit or quota exceeded.")

    app = build_app(settings, api_key="", embedder=Flaky())
    result = app.ingestion.ingest("p.pdf", sample_pdf)
    assert result.outcome == "failed" and "rate limit" in result.message
    assert app.store.count() == 0
    assert result.record.status == DocumentStatus.FAILED
    # a failed document can be retried (not treated as a duplicate)
    app_ok = build_app(settings, api_key="", embedder=HashingEmbedder())
    assert app_ok.ingestion.ingest("p.pdf", sample_pdf).outcome == "indexed"


def test_delete_document(app: DocuMindApp, sample_pdf: bytes, sample_docx: bytes) -> None:
    a = app.ingestion.ingest("a.pdf", sample_pdf).record.document_id
    b = app.ingestion.ingest("b.docx", sample_docx).record.document_id
    assert app.ingestion.delete(a)
    assert [r.document_id for r in app.ingestion.list_documents()] == [b]
    assert app.store.count(a) == 0 and app.store.count(b) > 0
    assert not app.ingestion.delete("doc_unknown")


def test_registry_persists_across_instances(settings: Settings, sample_pdf: bytes) -> None:
    from documind.embeddings.local_embeddings import HashingEmbedder

    doc_id = (
        build_app(settings, api_key="", embedder=HashingEmbedder())
        .ingestion.ingest("p.pdf", sample_pdf)
        .record.document_id
    )
    reopened = build_app(settings, api_key="", embedder=HashingEmbedder())
    assert reopened.ingestion.searchable_ids() == [doc_id]
    assert reopened.retriever.retrieve("hotel", [doc_id])


# ---- configuration ----------------------------------------------------------------
def test_missing_api_key_falls_back_to_offline_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = load_settings(use_dotenv=False, overrides={"chroma_persist_dir": tmp_path})
    assert not s.has_api_key
    assert s.resolved_embedding_provider() == "local"
    app = build_app(s)
    assert app.generator is None and app.embedder.provider == "local"


def test_forced_gemini_without_key_warns(tmp_path) -> None:
    s = Settings(embedding_provider="gemini", chroma_persist_dir=tmp_path)
    app = build_app(s, api_key="")
    assert app.embedder.provider == "local" and app.notices


def test_env_values_are_parsed(monkeypatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1000")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKEFAKEFAKEFAKEFAKEFAKE123")
    s = load_settings(use_dotenv=False)
    assert (s.top_k, s.chunk_size, s.gemini_model) == (7, 1000, "gemini-test-model")
    assert s.has_api_key and s.resolved_embedding_provider() == "gemini"
    assert "AIza" not in repr(s)  # key never appears in repr/logs


def test_invalid_overlap_config_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(chunk_size=300, chunk_overlap=300)


def test_secret_redaction() -> None:
    assert "AIza" not in redact("key=AIzaSyA1234567890abcdefghijklmnop failed")
    assert redact("api_key: abc123") == "api_key: [REDACTED]"
