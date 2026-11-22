"""Evaluation page: run the labelled benchmark and inspect real results."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from documind.app_factory import DocuMindApp, build_app
from documind.services.evaluation_service import EvaluationReport, evaluate, load_questions
from documind.ui import session
from documind.ui.theme import html, notice, page_header


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _run(app: DocuMindApp, answer_level: bool) -> EvaluationReport:
    tmp = Path(tempfile.mkdtemp(prefix="documind-eval-"))
    eval_app = build_app(
        app.settings,
        api_key="",
        persist_dir=tmp,
        embedder=app.embedder,
        generator=app.generator if answer_level else None,
    )
    ids: dict[str, str] = {}
    for path in sorted(session.DEMO_DIR.iterdir()):
        result = eval_app.ingestion.ingest(path.name, path.read_bytes())
        if result.record:
            ids[result.record.filename] = result.record.document_id
    return evaluate(
        load_questions(session.QUESTIONS_FILE),
        eval_app.retriever,
        ids,
        top_k=5,
        rag=eval_app.rag if answer_level else None,
    )


def _report(report: EvaluationReport) -> None:
    s, k = report.summary, report.top_k
    st.caption(
        f"Embeddings {report.embedding_key}. Answers: {report.generation_model or 'not evaluated'}. "
        f"Top-k {k}. Run at {report.created_at.replace('T', ' ')} UTC."
    )
    cols = st.columns(5)
    cols[0].metric(f"Recall@{k}", _fmt(s.get(f"recall_at_{k}")), help="Share of labelled evidence passages found")
    cols[1].metric(f"Hit rate@{k}", _fmt(s.get(f"hit_rate_at_{k}")), help="Questions with any evidence retrieved")
    cols[2].metric("MRR", _fmt(s.get("mrr")), help="Mean reciprocal rank of the first relevant passage")
    cols[3].metric("Sources found", _fmt(s.get("source_identification")), help="All expected documents in top-k")
    cols[4].metric(
        "Scope leaks", _fmt(s.get("out_of_scope_leaks")), help="Chunks from unselected documents (must be 0)"
    )
    st.caption(
        f"Recall@{k} by question language: English {_fmt(s.get(f'recall_at_{k}_en'))}, "
        f"Vietnamese {_fmt(s.get(f'recall_at_{k}_vi'))}."
    )
    if "citation_validity" in s:
        cols = st.columns(4)
        cols[0].metric("Citation validity", _fmt(s.get("citation_validity")))
        cols[1].metric("Cited evidence hit", _fmt(s.get("cited_evidence_hit")))
        cols[2].metric("Keyword match", _fmt(s.get("keyword_match")))
        cols[3].metric("Abstained when unanswerable", _fmt(s.get("unanswerable_handled")))

    st.markdown("#### Questions")
    st.dataframe(
        [
            {
                "ID": r.id,
                "Type": r.type,
                "Lang": r.language,
                "Question": r.question,
                f"Recall@{k}": r.recall_at_k,
                "First hit": r.first_relevant_rank,
                "Sources found": r.source_identified,
                "Citations valid": r.citation_valid,
                "Abstained": r.insufficient_evidence,
            }
            for r in report.results
        ],
        hide_index=True,
        width="stretch",
    )

    failures = [
        r
        for r in report.results
        if (r.answerable and (r.recall_at_k or 0) < 1.0)
        or r.leaked_out_of_scope
        or r.unanswerable_handled is False
        or r.citation_valid is False
        or r.error
    ]
    st.markdown(f"#### Failure cases ({len(failures)})")
    if not failures:
        st.caption("No failures in this run.")
    for r in failures:
        reasons = []
        if r.answerable and (r.recall_at_k or 0) < 1.0:
            reasons.append(f"recall {r.recall_at_k:.2f}")
        if r.leaked_out_of_scope:
            reasons.append("out-of-scope document retrieved")
        if r.unanswerable_handled is False:
            reasons.append("did not abstain")
        if r.citation_valid is False:
            reasons.append("fabricated citation removed")
        if r.error:
            reasons.append("generation error")
        with st.expander(f"{r.id}: {', '.join(reasons)}"):
            st.markdown(f"**{r.question}**  \n:small[{r.type}, {r.language}]")
            st.markdown("Retrieved, in rank order:")
            st.markdown("\n".join(f"{i}. {item}" for i, item in enumerate(r.retrieved, 1)) or "_nothing_")
            if r.answer:
                st.markdown("Answer:")
                st.text(r.answer)
            if r.error:
                notice(r.error, "err")


def render(app: DocuMindApp) -> None:
    page_header(
        "Evaluation",
        "Retrieval quality on 20 labelled questions over the sample documents. Every number here is measured.",
    )
    with st.expander("What is measured", icon=":material/info:"):
        st.markdown(
            """
Each answerable question in `demo/questions.json` names the exact evidence passage (document, phrase and page
or section). Retrieval metrics are deterministic string matches, not model judgements.

- **Recall@k**: share of expected evidence passages in the top-k chunks.
- **Hit rate@k**: questions with at least one evidence passage in the top-k.
- **MRR**: mean of 1 / rank of the first relevant chunk.
- **Sources found**: every expected document appears in the top-k.
- **Scope leaks**: chunks returned from documents outside the selected scope. Must be 0.
- **Answer checks** (optional, calls Gemini): citation validity, cited evidence, keyword match, abstention.
"""
        )

    saved = sorted(session.RESULTS_DIR.glob("*.json")) if session.RESULTS_DIR.exists() else []
    options = ["Run now", *[f"Saved: {p.stem}" for p in saved]]
    choice = st.segmented_control("Results", options, default="Run now", label_visibility="collapsed")

    if choice and choice.startswith("Saved: "):
        path = saved[options.index(choice) - 1]
        _report(EvaluationReport.model_validate_json(path.read_text(encoding="utf-8")))
        return

    html(
        f'<p class="dm-muted">Indexes the sample documents into a temporary store with the current embedder '
        f"({app.embedder.key}). Your knowledge base is not touched.</p>"
    )
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        run = st.button("Run evaluation", type="primary", icon=":material/play_arrow:")
        answer_level = st.checkbox(
            "Include answer checks (about 20 Gemini calls)",
            disabled=not app.generation_enabled,
            help=None if app.generation_enabled else "Requires a Gemini API key (see Settings).",
        )
    if run:
        with st.spinner("Indexing sample documents and running 20 questions…"):
            st.session_state.eval_report = _run(app, answer_level).model_dump(mode="json")
    if "eval_report" in st.session_state:
        report = EvaluationReport.model_validate(st.session_state.eval_report)
        _report(report)
        st.download_button(
            "Download results (JSON)",
            json.dumps(st.session_state.eval_report, ensure_ascii=False, indent=2),
            file_name="documind_evaluation.json",
            mime="application/json",
            icon=":material/download:",
            type="tertiary",
        )
