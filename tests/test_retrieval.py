"""ChromaDB storage, semantic search, document filtering and deletion."""

from pathlib import Path

import pytest

from documind.app_factory import DocuMindApp
from documind.embeddings.local_embeddings import HashingEmbedder
from documind.retrieval.retriever import Retriever
from documind.retrieval.vector_store import (
    IncompatibleCollectionError,
    VectorStore,
    collection_name_for,
    get_client,
)
from tests.conftest import make_docx, make_pdf


def _ingest(app: DocuMindApp, name: str, data: bytes) -> str:
    result = app.ingestion.ingest(name, data)
    assert result.outcome == "indexed", result.message
    return result.record.document_id


def test_semantic_search_returns_relevant_chunk(app: DocuMindApp, sample_pdf: bytes) -> None:
    doc_id = _ingest(app, "policy.pdf", sample_pdf)
    hits = app.retriever.retrieve("hotel reimbursement per night", [doc_id], top_k=2)
    assert hits[0].chunk.page_number == 2
    assert "USD 150" in hits[0].chunk.content
    assert hits[0].rank == 1
    assert -1.0 <= hits[0].similarity <= 1.0


def test_filtering_excludes_unselected_documents(app: DocuMindApp, sample_pdf: bytes) -> None:
    pdf_id = _ingest(app, "policy.pdf", sample_pdf)
    other = make_docx([("Leave", ["Employees are entitled to annual leave of thirty days."])])
    docx_id = _ingest(app, "other.docx", other)
    hits = app.retriever.retrieve("annual leave days", [docx_id], top_k=10)
    assert hits and {h.chunk.document_id for h in hits} == {docx_id}
    hits = app.retriever.retrieve("annual leave days", [pdf_id], top_k=10)
    assert {h.chunk.document_id for h in hits} == {pdf_id}


def test_empty_selection_and_empty_store(app: DocuMindApp, sample_pdf: bytes) -> None:
    assert app.retriever.retrieve("anything", [], top_k=5) == []
    assert app.retriever.retrieve("anything", ["doc_missing"], top_k=5) == []
    doc_id = _ingest(app, "policy.pdf", sample_pdf)
    assert app.retriever.retrieve("   ", [doc_id]) == []


def test_min_similarity_filters_weak_matches(app: DocuMindApp, sample_pdf: bytes) -> None:
    doc_id = _ingest(app, "policy.pdf", sample_pdf)
    strict = Retriever(app.embedder, app.store, min_similarity=0.99)
    assert strict.retrieve("completely unrelated quantum chromodynamics", [doc_id]) == []


def test_upsert_is_idempotent(app: DocuMindApp, sample_pdf: bytes) -> None:
    doc_id = _ingest(app, "policy.pdf", sample_pdf)
    before = app.store.count(doc_id)
    chunks = app.registry.load_chunks(doc_id)
    app.store.upsert(chunks, app.embedder.embed_documents([c.embedding_text() for c in chunks]))
    assert app.store.count(doc_id) == before


def test_delete_removes_vectors(app: DocuMindApp, sample_pdf: bytes) -> None:
    doc_id = _ingest(app, "policy.pdf", sample_pdf)
    assert app.store.count(doc_id) > 0
    assert app.ingestion.delete(doc_id)
    assert app.store.count(doc_id) == 0
    assert app.registry.get(doc_id) is None
    assert app.registry.load_chunks(doc_id) == []
    assert app.retriever.retrieve("hotel", [doc_id]) == []


def test_incompatible_embedding_space_is_detected(tmp_path: Path) -> None:
    key = "gemini:gemini-embedding-001:768"
    client = get_client(tmp_path)
    # Simulate a collection whose recorded embedding space differs from ours.
    client.create_collection(
        collection_name_for(key),
        metadata={"hnsw:space": "cosine", "embedding_key": "other:model:3072", "dimension": 3072},
        embedding_function=None,
    )
    with pytest.raises(IncompatibleCollectionError):
        VectorStore(tmp_path, key, 768)


def test_vector_dimension_mismatch_on_upsert(app: DocuMindApp, sample_pdf: bytes) -> None:
    doc_id = _ingest(app, "policy.pdf", sample_pdf)
    chunks = app.registry.load_chunks(doc_id)[:1]
    with pytest.raises(IncompatibleCollectionError):
        app.store.upsert(chunks, [[0.1] * 10])


def test_retriever_refuses_mismatched_embedder(app: DocuMindApp) -> None:
    with pytest.raises(ValueError):
        Retriever(HashingEmbedder(dimension=128), app.store)


def test_model_change_requires_reindex(settings, sample_pdf: bytes) -> None:
    from documind.app_factory import build_app

    app1 = build_app(settings, api_key="", embedder=HashingEmbedder(dimension=1024))
    doc_id = _ingest(app1, "policy.pdf", sample_pdf)
    app2 = build_app(settings, api_key="", embedder=HashingEmbedder(dimension=512))
    record = app2.registry.get(doc_id)
    assert app2.ingestion.needs_reindex(record)
    assert app2.ingestion.searchable_ids() == []
    assert app2.retriever.retrieve("hotel", [doc_id]) == []  # old vectors never queried
    app2.ingestion.reindex(doc_id)
    assert app2.ingestion.searchable_ids() == [doc_id]
    assert app2.retriever.retrieve("hotel per night", [doc_id])


def test_pdf_with_many_pages_keeps_page_numbers(app: DocuMindApp) -> None:
    data = make_pdf([[f"Topic {i}: the code word for this page is marker{i}."] for i in range(1, 6)])
    doc_id = _ingest(app, "pages.pdf", data)
    hit = app.retriever.retrieve("marker4", [doc_id], top_k=1)[0]
    assert hit.chunk.page_number == 4
