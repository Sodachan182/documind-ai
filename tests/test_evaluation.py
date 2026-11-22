"""Evaluation metrics on the real demo dataset (offline lexical embeddings)."""

from pathlib import Path

from documind.app_factory import DocuMindApp
from documind.models import Chunk, FileType
from documind.services.evaluation_service import EvidenceRef, evaluate, load_questions, matches
from tests.conftest import DEMO_DIR, FakeGenerator

QUESTIONS = Path(__file__).resolve().parents[1] / "demo" / "questions.json"


def _index_demo(app: DocuMindApp) -> dict[str, str]:
    ids = {}
    for path in sorted(DEMO_DIR.iterdir()):
        record = app.ingestion.ingest(path.name, path.read_bytes()).record
        ids[record.filename] = record.document_id
    return ids


def test_question_set_is_well_formed() -> None:
    qs = load_questions(QUESTIONS)
    assert len(qs) >= 15
    types = {q.type for q in qs}
    assert {"single", "multi", "comparison", "conflict", "unanswerable", "isolation"} <= types
    assert {q.language for q in qs} == {"en", "vi"}
    assert len({q.id for q in qs}) == len(qs)


def test_every_evidence_reference_exists_in_the_demo_corpus(app: DocuMindApp) -> None:
    ids = _index_demo(app)
    chunks = [c for doc_id in ids.values() for c in app.registry.load_chunks(doc_id)]
    for q in load_questions(QUESTIONS):
        for ev in q.evidence:
            assert any(matches(c, ev) for c in chunks), f"{q.id}: evidence not found: {ev}"


def test_matches_requires_document_page_and_phrase() -> None:
    chunk = Chunk(
        chunk_id="a",
        document_id="d",
        filename="H.pdf",
        file_hash="h",
        file_type=FileType.PDF,
        content="Employees get 18  Working days.",
        chunk_index=0,
        page_number=3,
    )
    assert matches(chunk, EvidenceRef(document="H.pdf", page=3, contains="18 working days"))
    assert not matches(chunk, EvidenceRef(document="H.pdf", page=4, contains="18 working days"))
    assert not matches(chunk, EvidenceRef(document="X.pdf", contains="18 working days"))


def test_retrieval_metrics_on_demo(app: DocuMindApp) -> None:
    ids = _index_demo(app)
    report = evaluate(load_questions(QUESTIONS), app.retriever, ids, top_k=5)
    s = report.summary
    assert s["questions"] == 20
    assert 0.0 <= s["recall_at_5"] <= 1.0 and 0.0 <= s["mrr"] <= 1.0
    assert s["out_of_scope_leaks"] == 0  # document isolation holds
    single_en = [r for r in report.results if r.id in {"Q01", "Q02", "Q04"}]
    assert all(r.first_relevant_rank == 1 for r in single_en)
    iso = next(r for r in report.results if r.type == "isolation")
    assert all(item.startswith("Employee Handbook.pdf") for item in iso.retrieved)


def test_answer_level_metrics_with_test_double(app: DocuMindApp) -> None:
    ids = _index_demo(app)
    qs = [q for q in load_questions(QUESTIONS) if q.id in {"Q01", "Q16"}]
    rag = app.rag
    rag.generator = FakeGenerator("Full-time employees receive 18 working days [S1] [S99].")
    report = evaluate(qs, app.retriever, ids, top_k=5, rag=rag)
    q01 = next(r for r in report.results if r.id == "Q01")
    assert q01.keyword_match and q01.cited_evidence_hit
    assert q01.citation_valid is False  # S99 was fabricated
    q16 = next(r for r in report.results if r.id == "Q16")
    assert q16.unanswerable_handled is False  # the double never flags insufficiency
