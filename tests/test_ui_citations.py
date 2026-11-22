"""Pure rendering helpers of the UI (no Streamlit runtime needed)."""

from documind.models import Chunk, Citation, FileType, RAGAnswer
from documind.ui.citations import _sources_html, answer_markdown, numbering


def _src(sid: str, page: int) -> Citation:
    chunk = Chunk(
        chunk_id=f"d:{page}",
        document_id="d",
        filename="H.pdf",
        file_hash="h",
        file_type=FileType.PDF,
        content=f"<b>passage {page}</b>",
        chunk_index=page,
        page_number=page,
    )
    return Citation(source_id=sid, chunk=chunk, similarity=0.5)


def _answer(cited: list[str]) -> RAGAnswer:
    return RAGAnswer(
        question="q",
        answer="x",
        language="en",
        mode="generated",
        sources=[_src("S1", 1), _src("S2", 2), _src("S3", 3)],
        cited_ids=cited,
    )


def test_numbering_follows_citation_order_then_retrieval_order() -> None:
    assert numbering(_answer(["S2", "S1"])) == {"S2": 1, "S1": 2, "S3": 3}


def test_markers_become_badges_and_dollars_are_escaped() -> None:
    nums = {"S2": 1, "S1": 2}
    out = answer_markdown("Cost is $25 [S2]. Also [S1][S9].", nums)
    assert out == r"Cost is \$25 :primary-badge[1]. Also :primary-badge[2]."


def test_sources_listed_in_number_order_and_escaped() -> None:
    answer = _answer(["S2", "S1"])
    html = _sources_html(answer.sources, numbering(answer), set(answer.cited_ids))
    assert html.index("Page 2") < html.index("Page 1") < html.index("Page 3")
    assert "<b>passage" not in html and "&lt;b&gt;passage 1&lt;/b&gt;" in html
    assert "not cited" in html
