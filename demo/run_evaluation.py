"""Run the retrieval (and optionally answer) evaluation on the demo dataset.

    python demo/run_evaluation.py                 # offline: lexical embeddings, retrieval only
    python demo/run_evaluation.py --mode gemini   # Gemini embeddings + generation (needs GEMINI_API_KEY)

The demo documents are indexed into a fresh temporary ChromaDB directory, so
your own knowledge base is never touched. Results are written to
``docs/evaluation_results/<mode>.json``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.app_factory import build_app  # noqa: E402
from documind.config import load_settings  # noqa: E402
from documind.services.evaluation_service import evaluate, load_questions  # noqa: E402

DOCS_DIR = ROOT / "demo" / "documents"
OUT_DIR = ROOT / "docs" / "evaluation_results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "gemini"], default="offline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between Gemini calls")
    args = parser.parse_args()

    settings = load_settings()
    if args.mode == "gemini" and not settings.has_api_key:
        print("GEMINI_API_KEY is not set; cannot run the Gemini evaluation.", file=sys.stderr)
        return 2
    provider = "gemini" if args.mode == "gemini" else "local"
    settings = settings.model_copy(update={"embedding_provider": provider})
    api_key = settings.gemini_api_key if args.mode == "gemini" else ""

    with tempfile.TemporaryDirectory(prefix="documind-eval-", ignore_cleanup_errors=True) as tmp:
        app = build_app(settings, api_key=api_key, persist_dir=Path(tmp))
        ids: dict[str, str] = {}
        for path in sorted(DOCS_DIR.iterdir()):
            result = app.ingestion.ingest(path.name, path.read_bytes())
            print(f"[{result.outcome}] {path.name}: {result.message}")
            if result.record and result.outcome in ("indexed", "duplicate"):
                ids[result.record.filename] = result.record.document_id
        questions = load_questions(ROOT / "demo" / "questions.json")
        report = evaluate(
            questions,
            app.retriever,
            ids,
            top_k=args.top_k,
            rag=app.rag if args.mode == "gemini" else None,
            delay_s=args.delay,
        )
        del app  # release Chroma handles before the temp dir is removed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.mode}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nEmbedding space: {report.embedding_key}   generation: {report.generation_model or 'none'}")
    for key, value in report.summary.items():
        print(f"  {key:28s} {value}")
    print("\n| ID | Type | Lang | Recall@k | RR | First relevant rank |")
    print("|---|---|---|---|---|---|")
    for r in report.results:
        rec = "-" if r.recall_at_k is None else f"{r.recall_at_k:.2f}"
        rr = "-" if r.reciprocal_rank is None else f"{r.reciprocal_rank:.2f}"
        print(f"| {r.id} | {r.type} | {r.language} | {rec} | {rr} | {r.first_relevant_rank or '-'} |")
    print(f"\nSaved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
