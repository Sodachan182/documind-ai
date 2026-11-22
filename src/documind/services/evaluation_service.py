"""Retrieval and answer evaluation against a labelled question set.

Deterministic metrics (no LLM judge):
  * Recall@k  - share of expected evidence passages found in the top-k chunks.
  * Hit@k     - 1 if any expected evidence passage is in the top-k.
  * MRR       - mean of 1/rank of the first relevant chunk (0 if none).
  * Source identification - all expected documents appear in the top-k.
Answer-level metrics (only when a generator is configured):
  * Citation validity - the answer contained no fabricated citation IDs.
  * Cited-evidence hit - at least one cited source is a labelled evidence passage.
  * Keyword match - all expected answer keywords appear in the answer (string match).
  * Unanswerable handling - the model flagged insufficient evidence.
"""

from __future__ import annotations

import json
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field

from documind.models import Chunk
from documind.retrieval.retriever import Retriever
from documind.services.rag_service import RAGService

QuestionType = Literal["single", "multi", "comparison", "conflict", "unanswerable", "isolation"]


class EvidenceRef(BaseModel):
    document: str
    contains: str
    page: int | None = None
    section: str | None = None


class EvalQuestion(BaseModel):
    id: str
    question: str
    language: Literal["en", "vi"]
    type: QuestionType
    expected_documents: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    answer_keywords: list[str] = Field(default_factory=list)
    scope: list[str] | None = None  # restrict retrieval to these documents
    notes: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


class QuestionResult(BaseModel):
    id: str
    question: str
    type: QuestionType
    language: str
    answerable: bool
    retrieved: list[str]  # "filename | location" for each rank
    first_relevant_rank: int | None = None
    recall_at_k: float | None = None
    hit_at_k: float | None = None
    reciprocal_rank: float | None = None
    source_identified: bool | None = None
    leaked_out_of_scope: bool = False
    # answer-level (None when no generator)
    answer: str | None = None
    insufficient_evidence: bool | None = None
    citation_valid: bool | None = None
    cited_evidence_hit: bool | None = None
    keyword_match: bool | None = None
    unanswerable_handled: bool | None = None
    error: str | None = None


class EvaluationReport(BaseModel):
    created_at: str
    embedding_key: str
    generation_model: str | None
    top_k: int
    num_questions: int
    summary: dict[str, float | int | None]
    results: list[QuestionResult]


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def matches(chunk: Chunk, ev: EvidenceRef) -> bool:
    if chunk.filename != ev.document:
        return False
    if ev.page is not None and chunk.page_number != ev.page:
        return False
    if ev.section is not None and chunk.section_heading != ev.section:
        return False
    return _norm(ev.contains) in _norm(chunk.content)


def load_questions(path: Path | str) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    return [EvalQuestion.model_validate(q) for q in items]


def _avg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def evaluate(
    questions: list[EvalQuestion],
    retriever: Retriever,
    document_ids_by_name: dict[str, str],
    *,
    top_k: int = 5,
    rag: RAGService | None = None,
    delay_s: float = 0.0,
) -> EvaluationReport:
    results: list[QuestionResult] = []
    all_ids = list(document_ids_by_name.values())
    for q in questions:
        scope_names = q.scope or list(document_ids_by_name)
        scope_ids = [document_ids_by_name[n] for n in scope_names if n in document_ids_by_name]
        retrieved = retriever.retrieve(q.question, scope_ids or [], top_k)
        chunks = [r.chunk for r in retrieved]
        res = QuestionResult(
            id=q.id,
            question=q.question,
            type=q.type,
            language=q.language,
            answerable=q.answerable,
            retrieved=[f"{c.filename} | {c.location_label}" for c in chunks],
            leaked_out_of_scope=any(c.filename not in scope_names for c in chunks),
        )
        if q.answerable:
            found = [any(matches(c, ev) for c in chunks) for ev in q.evidence]
            ranks = [i for i, c in enumerate(chunks, start=1) if any(matches(c, ev) for ev in q.evidence)]
            res.recall_at_k = sum(found) / len(found)
            res.hit_at_k = 1.0 if any(found) else 0.0
            res.first_relevant_rank = ranks[0] if ranks else None
            res.reciprocal_rank = 1.0 / ranks[0] if ranks else 0.0
            res.source_identified = set(q.expected_documents) <= {c.filename for c in chunks}

        if rag is not None and rag.generator is not None:
            ids = scope_ids if q.scope else all_ids
            answer = rag.ask(q.question, ids, top_k)
            if answer.mode == "error":
                res.error = answer.answer
            else:
                res.answer = answer.answer
                res.insufficient_evidence = answer.insufficient_evidence
                res.citation_valid = not answer.invalid_citations
                if q.answerable:
                    cited = [s.chunk for s in answer.cited_sources]
                    res.cited_evidence_hit = any(matches(c, ev) for c in cited for ev in q.evidence)
                    text = _norm(answer.answer)
                    res.keyword_match = all(_norm(k) in text for k in q.answer_keywords)
                else:
                    res.unanswerable_handled = answer.insufficient_evidence
            if delay_s:
                time.sleep(delay_s)
        results.append(res)

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    summary: dict[str, float | int | None] = {
        "questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        f"recall_at_{top_k}": _avg([r.recall_at_k for r in answerable]),
        f"hit_rate_at_{top_k}": _avg([r.hit_at_k for r in answerable]),
        "mrr": _avg([r.reciprocal_rank for r in answerable]),
        "source_identification": _avg([1.0 if r.source_identified else 0.0 for r in answerable]),
        "out_of_scope_leaks": sum(r.leaked_out_of_scope for r in results),
    }
    for lang in ("en", "vi"):
        subset = [r for r in answerable if r.language == lang]
        summary[f"recall_at_{top_k}_{lang}"] = _avg([r.recall_at_k for r in subset])
    if rag is not None and rag.generator is not None:
        answered = [r for r in results if r.error is None]
        summary.update(
            {
                "generation_errors": sum(r.error is not None for r in results),
                "citation_validity": _avg([1.0 if r.citation_valid else 0.0 for r in answered]),
                "cited_evidence_hit": _avg([1.0 if r.cited_evidence_hit else 0.0 for r in answered if r.answerable]),
                "keyword_match": _avg([1.0 if r.keyword_match else 0.0 for r in answered if r.answerable]),
                "unanswerable_handled": _avg(
                    [1.0 if r.unanswerable_handled else 0.0 for r in answered if not r.answerable]
                ),
            }
        )
    generator = rag.generator if rag is not None else None
    return EvaluationReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        embedding_key=retriever.embedder.key,
        generation_model=getattr(generator, "model", None),
        top_k=top_k,
        num_questions=len(results),
        summary=summary,
        results=results,
    )
