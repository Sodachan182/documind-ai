import pymupdf
import pytest

from documind.document_processing.pdf_reader import DocumentExtractionError, extract_pdf
from documind.models import FileType
from tests.conftest import DEMO_DIR, make_pdf


def test_extracts_text_with_exact_page_numbers(sample_pdf: bytes) -> None:
    doc = extract_pdf(sample_pdf, "policy.pdf")
    assert doc.file_type == FileType.PDF
    assert doc.page_count == 2
    assert not doc.needs_ocr
    pages = {b.page_number for b in doc.blocks}
    assert pages == {1, 2}
    leave = [b for b in doc.blocks if "twelve days" in b.text]
    assert leave and leave[0].page_number == 1
    hotel = [b for b in doc.blocks if "USD 150" in b.text]
    assert hotel and hotel[0].page_number == 2


def test_blank_pdf_is_flagged_for_ocr() -> None:
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    result = extract_pdf(data, "scan.pdf")
    assert result.needs_ocr
    assert result.blocks == []
    assert "OCR" in result.warnings[0]


def test_partially_empty_pdf_warns_about_pages() -> None:
    data = make_pdf([["A full page of normal policy text that is long enough to count. " * 3], []])
    result = extract_pdf(data, "mixed.pdf")
    assert not result.needs_ocr
    assert any("page(s) 2" in w for w in result.warnings)


def test_corrupted_pdf_raises() -> None:
    with pytest.raises(DocumentExtractionError):
        extract_pdf(b"%PDF-1.7 this is not really a pdf", "broken.pdf")


def test_unicode_vietnamese_text_is_preserved() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    # Built-in PDF fonts lack Vietnamese glyphs; use the bundled Noto-style font via insert_htmlbox.
    page.insert_htmlbox(pymupdf.Rect(72, 72, 500, 300), "<p>Nhân viên được nghỉ phép 18 ngày mỗi năm.</p>")
    result = extract_pdf(doc.tobytes(), "vi.pdf")
    text = " ".join(b.text for b in result.blocks)
    assert "Nhân viên được nghỉ phép 18 ngày" in text


def test_demo_handbook_pages_match_ground_truth() -> None:
    data = (DEMO_DIR / "Employee Handbook.pdf").read_bytes()
    result = extract_pdf(data, "Employee Handbook.pdf")
    assert result.page_count == 7
    page_of = {b.text[:20]: b.page_number for b in result.blocks}
    annual = next(b for b in result.blocks if "18 working days" in b.text)
    assert annual.page_number == 3
    assert page_of  # sanity
