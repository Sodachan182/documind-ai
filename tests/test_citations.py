from documind.citations.citation_manager import build_citations, validate_citations
from documind.models import Chunk, FileType, RetrievedChunk


def _retrieved(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"d:{i}",
                document_id="d",
                filename="h.pdf",
                file_hash="x",
                file_type=FileType.PDF,
                content=f"passage {i}",
                chunk_index=i,
                page_number=i + 1,
            ),
            similarity=0.9 - i * 0.1,
            rank=i + 1,
        )
        for i in range(n)
    ]


def test_ids_are_assigned_by_application_in_rank_order() -> None:
    sources = build_citations(_retrieved(3))
    assert [s.source_id for s in sources] == ["S1", "S2", "S3"]
    assert sources[1].chunk.page_number == 2


def test_valid_citations_are_kept_and_mapped() -> None:
    sources = build_citations(_retrieved(3))
    check = validate_citations("Leave is 18 days [S1]. Sick leave is 10 days [S3].", sources)
    assert check.valid_ids == ["S1", "S3"]
    assert check.invalid_ids == []
    assert check.text == "Leave is 18 days [S1]. Sick leave is 10 days [S3]."


def test_fabricated_citations_are_removed() -> None:
    sources = build_citations(_retrieved(2))
    check = validate_citations("Fact A [S1]. Fact B [S7]. Fact C [S2, S9].", sources)
    assert check.valid_ids == ["S1", "S2"]
    assert check.invalid_ids == ["S7", "S9"]
    assert "S7" not in check.text and "S9" not in check.text
    assert check.text == "Fact A [S1]. Fact B. Fact C [S2]."


def test_grouped_and_variant_formats_are_normalised() -> None:
    sources = build_citations(_retrieved(3))
    check = validate_citations("Both apply [S1; s3] and 【S2】.", sources)
    assert check.valid_ids == ["S1", "S3", "S2"]
    assert check.text == "Both apply [S1][S3] and [S2]."


def test_text_without_citations() -> None:
    check = validate_citations("No sources here.", build_citations(_retrieved(1)))
    assert check.valid_ids == [] and check.invalid_ids == []


def test_docx_location_label_has_no_page() -> None:
    chunk = Chunk(
        chunk_id="p:0",
        document_id="p",
        filename="Policy.docx",
        file_hash="y",
        file_type=FileType.DOCX,
        content="x",
        chunk_index=0,
        section_heading="Access Control",
        paragraph_index=14,
    )
    assert chunk.location_label == "Section: Access Control · Paragraph 14"
    assert "Page" not in chunk.location_label
