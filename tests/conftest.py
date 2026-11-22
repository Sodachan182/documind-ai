"""Shared fixtures. All offline: no network, no API key."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest
from docx import Document

from documind.app_factory import DocuMindApp, build_app
from documind.config import Settings
from documind.embeddings.local_embeddings import HashingEmbedder
from documind.llm.gemini_client import GeneratedAnswer
from documind.models import Citation, TokenUsage

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo" / "documents"


def make_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a PDF where each page contains the given paragraphs."""
    doc = pymupdf.open()
    for paragraphs in pages:
        page = doc.new_page()
        y = 72
        for para in paragraphs:
            rect = pymupdf.Rect(72, y, 520, y + 120)
            page.insert_textbox(rect, para, fontsize=11)
            y += 130
    data = doc.tobytes()
    doc.close()
    return data


def make_docx(sections: Sequence[tuple[str, Sequence[str | list[list[str]]]]]) -> bytes:
    doc = Document()
    for heading, items in sections:
        doc.add_heading(heading, level=1)
        for item in items:
            if isinstance(item, list):
                table = doc.add_table(rows=len(item), cols=len(item[0]))
                for r, row in enumerate(item):
                    for c, value in enumerate(row):
                        table.cell(r, c).text = value
            else:
                doc.add_paragraph(item)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class FakeGenerator:
    """TEST DOUBLE ONLY - returns a scripted answer; never used by the app."""

    model = "fake-generator (test double)"

    def __init__(self, answer: str = "", insufficient: bool = False) -> None:
        self.answer = answer
        self.insufficient = insufficient
        self.calls: list[tuple[str, list[Citation], str]] = []

    def generate(self, question: str, sources: Sequence[Citation], language: str) -> GeneratedAnswer:
        self.calls.append((question, list(sources), language))
        return GeneratedAnswer(
            answer=self.answer,
            insufficient_evidence=self.insufficient,
            usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
            model=self.model,
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(chroma_persist_dir=tmp_path / "chroma", embedding_provider="local")


@pytest.fixture
def app(settings: Settings) -> DocuMindApp:
    return build_app(settings, api_key="", embedder=HashingEmbedder())


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf(
        [
            ["Leave Policy", "Full-time employees are entitled to twelve days of annual leave per year."],
            ["Travel Policy", "Hotel costs are reimbursed up to USD 150 per night with receipts."],
        ]
    )


@pytest.fixture
def sample_docx() -> bytes:
    return make_docx(
        [
            ("Access Control", ["Administrators must use hardware security keys.", "Access is reviewed monthly."]),
            ("Classification", ["Data levels are listed below.", [["Level", "Meaning"], ["Secret", "Board only"]]]),
        ]
    )
