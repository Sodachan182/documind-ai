"""Live end-to-end tests against the real Gemini API.

Excluded by default (``-m 'not live'`` in pyproject). Run with:
    GEMINI_API_KEY=... pytest -m live
"""

import os

import pytest

from documind.app_factory import build_app
from documind.config import load_settings
from tests.conftest import DEMO_DIR

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"),
]


@pytest.fixture(scope="module")
def live_app(tmp_path_factory):
    settings = load_settings(overrides={"chroma_persist_dir": tmp_path_factory.mktemp("live")})
    app = build_app(settings)
    ids = {}
    for path in sorted(DEMO_DIR.iterdir()):
        record = app.ingestion.ingest(path.name, path.read_bytes()).record
        assert record.status.value == "indexed", record.error
        ids[record.filename] = record.document_id
    return app, ids


def test_live_grounded_answer_with_citation(live_app) -> None:
    app, ids = live_app
    answer = app.rag.ask("How many days of paid annual leave do full-time employees get?", list(ids.values()))
    assert answer.mode == "generated", answer.answer
    assert "18" in answer.answer
    assert answer.cited_ids and not answer.invalid_citations
    assert any(s.chunk.page_number == 3 for s in answer.cited_sources)


def test_live_vietnamese_question(live_app) -> None:
    app, ids = live_app
    answer = app.rag.ask("Nhân viên được làm việc từ xa tối đa bao nhiêu ngày mỗi tuần?", list(ids.values()))
    assert answer.language == "vi" and "3" in answer.answer


def test_live_unanswerable(live_app) -> None:
    app, ids = live_app
    answer = app.rag.ask("What is the company's parental leave policy?", list(ids.values()))
    assert answer.insufficient_evidence
