"""Live ShopAIKey checks (real API, uses a few tokens of credit).

Excluded by default (``-m 'not live'``). Runs only when SHOPAIKEY_API_KEY and
SHOPAIKEY_MODEL are set; SHOPAIKEY_PROTOCOL selects the adapter (default openai):

    SHOPAIKEY_API_KEY=sk-... SHOPAIKEY_MODEL=<model> pytest -m live tests/test_live_shopaikey.py

Exactly two short requests: one connection check and one grounded RAG answer.
"""

import os

import pytest

from documind.app_factory import GenerationSelection, build_app
from documind.config import load_settings
from documind.embeddings.local_embeddings import HashingEmbedder
from tests.conftest import DEMO_DIR

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("SHOPAIKEY_API_KEY") and os.environ.get("SHOPAIKEY_MODEL")),
        reason="SHOPAIKEY_API_KEY and SHOPAIKEY_MODEL not set",
    ),
]


@pytest.fixture(scope="module")
def live_app(tmp_path_factory):
    settings = load_settings(overrides={"chroma_persist_dir": tmp_path_factory.mktemp("live_sak")})
    selection = GenerationSelection(
        provider="shopaikey",
        shopaikey_key=settings.shopaikey_api_key,
        shopaikey_protocol=settings.shopaikey_protocol,
        shopaikey_model=settings.shopaikey_model,
        shopaikey_root=settings.shopaikey_base_url,
    )
    # Offline embeddings keep this test to two generation requests.
    app = build_app(settings, embedder=HashingEmbedder(), selection=selection)
    path = DEMO_DIR / "Employee Handbook.pdf"
    doc_id = app.ingestion.ingest(path.name, path.read_bytes()).record.document_id
    return app, doc_id


def test_live_connection_check(live_app) -> None:
    app, _ = live_app
    assert app.generator.check()


def test_live_grounded_answer(live_app) -> None:
    app, doc_id = live_app
    answer = app.rag.ask("How many days of paid annual leave do full-time employees receive?", [doc_id])
    assert answer.mode == "generated", answer.answer
    assert "18" in answer.answer and answer.cited_ids and not answer.invalid_citations
