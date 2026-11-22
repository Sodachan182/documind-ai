import pytest

from documind.document_processing.docx_reader import extract_docx
from documind.document_processing.pdf_reader import DocumentExtractionError
from documind.models import FileType
from tests.conftest import DEMO_DIR, make_docx


def test_paragraphs_keep_section_and_paragraph_index(sample_docx: bytes) -> None:
    doc = extract_docx(sample_docx, "policy.docx")
    assert doc.file_type == FileType.DOCX
    assert doc.page_count is None  # never invent page numbers for DOCX
    keys = next(b for b in doc.blocks if "hardware security keys" in b.text)
    assert keys.section_heading == "Access Control"
    assert keys.paragraph_index == 2  # heading is paragraph 1
    assert keys.page_number is None
    monthly = next(b for b in doc.blocks if "reviewed monthly" in b.text)
    assert monthly.paragraph_index == 3


def test_tables_are_extracted_in_order(sample_docx: bytes) -> None:
    doc = extract_docx(sample_docx, "policy.docx")
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.section_heading == "Classification"
    assert table.table_index == 1
    assert "Level | Meaning" in table.text
    assert "Secret | Board only" in table.text
    kinds = [b.kind for b in doc.blocks]
    assert kinds.index("table") == len(kinds) - 1


def test_vietnamese_docx_text() -> None:
    data = make_docx([("Chính sách", ["Mật khẩu phải có ít nhất 14 ký tự."])])
    doc = extract_docx(data, "vi.docx")
    assert doc.blocks[0].text == "Mật khẩu phải có ít nhất 14 ký tự."
    assert doc.blocks[0].section_heading == "Chính sách"


def test_empty_docx_warns() -> None:
    doc = extract_docx(make_docx([]), "empty.docx")
    assert doc.blocks == []
    assert doc.warnings


def test_corrupted_docx_raises() -> None:
    with pytest.raises(DocumentExtractionError):
        extract_docx(b"PK\x03\x04garbage", "bad.docx")


def test_demo_policy_sections() -> None:
    data = (DEMO_DIR / "Company Information Security Policy.docx").read_bytes()
    doc = extract_docx(data, "Company Information Security Policy.docx")
    pw = next(b for b in doc.blocks if "at least 14 characters" in b.text)
    assert pw.section_heading == "Password Requirements"
    retention = next(b for b in doc.blocks if "Audit logs | 24 months" in b.text)
    assert retention.kind == "table" and retention.section_heading == "Retention Requirements"
