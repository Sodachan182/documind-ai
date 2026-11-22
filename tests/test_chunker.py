import itertools
import unicodedata

import pytest

from documind.document_processing.chunker import chunk_document
from documind.document_processing.cleaner import clean_paragraph, normalize_text
from documind.models import ExtractedDocument, FileType, TextBlock


def _pdf(blocks: list[TextBlock]) -> ExtractedDocument:
    return ExtractedDocument(filename="a.pdf", file_type=FileType.PDF, blocks=blocks, page_count=3)


def _chunk(doc: ExtractedDocument, size: int = 300, overlap: int = 50):
    return chunk_document(doc, document_id="doc_x", file_hash="h" * 64, chunk_size=size, chunk_overlap=overlap)


# ---- cleaner -------------------------------------------------------------------
def test_normalize_composes_vietnamese_and_removes_controls() -> None:
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    out = normalize_text(f"  {decomposed}\x00​  text\r\n\r\n\r\nnext  ")
    assert out == "Tiếng Việt text\n\nnext"
    assert unicodedata.is_normalized("NFC", out)


def test_clean_paragraph_joins_wrapped_lines_and_ligatures() -> None:
    assert clean_paragraph("The ofﬁce is\nopen to every-\none") == "The office is open to everyone"


# ---- chunker -------------------------------------------------------------------
def test_chunks_never_cross_pages() -> None:
    blocks = [TextBlock(text=f"Page one sentence {i}. " * 5, page_number=1) for i in range(5)]
    blocks += [TextBlock(text=f"Page two sentence {i}. " * 5, page_number=2) for i in range(5)]
    chunks = _chunk(_pdf(blocks))
    for c in chunks:
        assert c.page_number in (1, 2)
        if c.page_number == 1:
            assert "Page two" not in c.content
        else:
            assert "Page one" not in c.content


def test_chunk_size_respected_and_overlap_present() -> None:
    text = " ".join(f"Sentence number {i} has some words." for i in range(60))
    chunks = _chunk(_pdf([TextBlock(text=text, page_number=1)]), size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.content) <= 300 for c in chunks)
    for prev, nxt in itertools.pairwise(chunks):
        tail_words = prev.content.split()[-3:]
        assert " ".join(tail_words) in nxt.content  # overlap carries context forward


def test_metadata_and_stable_ids() -> None:
    blocks = [TextBlock(text="Alpha. " * 10, page_number=2)]
    first = _chunk(_pdf(blocks))
    second = _chunk(_pdf(blocks))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    c = first[0]
    assert c.chunk_id == "doc_x:0000"
    assert (c.document_id, c.filename, c.file_hash, c.page_number, c.chunk_index) == (
        "doc_x",
        "a.pdf",
        "h" * 64,
        2,
        0,
    )
    assert c.location_label == "Page 2"


def test_docx_chunks_track_section_and_paragraph_range() -> None:
    doc = ExtractedDocument(
        filename="p.docx",
        file_type=FileType.DOCX,
        blocks=[
            TextBlock(text="First rule.", section_heading="Access", paragraph_index=2),
            TextBlock(text="Second rule.", section_heading="Access", paragraph_index=3),
            TextBlock(text="Other topic.", section_heading="Retention", paragraph_index=5),
            TextBlock(text="A | B", kind="table", section_heading="Retention", table_index=1),
        ],
    )
    chunks = _chunk(doc)
    assert [c.section_heading for c in chunks] == ["Access", "Retention", "Retention"]
    assert chunks[0].paragraph_index == 2 and chunks[0].paragraph_end == 3
    assert chunks[0].location_label == "Section: Access · Paragraphs 2-3"
    assert chunks[1].location_label == "Section: Retention · Paragraph 5"
    assert chunks[2].table_index == 1 and "Table 1" in chunks[2].location_label
    assert all(c.page_number is None for c in chunks)


def test_oversized_token_is_hard_split() -> None:
    chunks = _chunk(_pdf([TextBlock(text="x" * 1000, page_number=1)]), size=300, overlap=50)
    assert all(len(c.content) <= 300 for c in chunks)
    assert "".join(c.content.replace("\n", "") for c in chunks).count("x") >= 1000


def test_empty_document_yields_no_chunks() -> None:
    assert _chunk(_pdf([])) == []


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        _chunk(_pdf([]), size=100, overlap=100)
