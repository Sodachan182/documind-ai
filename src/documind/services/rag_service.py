"""The RAG pipeline: validate -> retrieve -> prompt -> generate -> validate citations."""

from __future__ import annotations

import time
from collections.abc import Sequence

from documind.citations.citation_manager import build_citations, validate_citations
from documind.embeddings.base import EmbeddingError
from documind.llm.errors import GenerationError
from documind.llm.gemini_client import AnswerGenerator
from documind.llm.prompts import INSUFFICIENT_EVIDENCE_MESSAGES, detect_language
from documind.models import RAGAnswer
from documind.retrieval.retriever import Retriever

NO_DOCS_MESSAGES = {
    "en": "No documents are selected. Select at least one indexed document in the sidebar.",
    "vi": "Chưa chọn tài liệu nào. Hãy chọn ít nhất một tài liệu đã được lập chỉ mục ở thanh bên.",
}
RETRIEVAL_ONLY_MESSAGES = {
    "en": "Gemini is not configured, so no answer was generated. "
    "The most relevant passages retrieved from your documents are shown below.",
    "vi": "Chưa cấu hình Gemini nên không có câu trả lời được tạo. "
    "Các đoạn liên quan nhất được truy xuất từ tài liệu được hiển thị bên dưới.",
}


class QuestionValidationError(ValueError):
    pass


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator | None,
        *,
        max_question_chars: int = 2000,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.max_question_chars = max_question_chars

    def validate_question(self, question: str) -> str:
        cleaned = " ".join((question or "").split())
        if not cleaned:
            raise QuestionValidationError("Please enter a question.")
        if len(cleaned) > self.max_question_chars:
            raise QuestionValidationError(
                f"The question is too long ({len(cleaned)} characters; max {self.max_question_chars})."
            )
        return cleaned

    def ask(self, question: str, document_ids: Sequence[str], top_k: int | None = None) -> RAGAnswer:
        started = time.perf_counter()
        q = self.validate_question(question)
        lang = detect_language(q)

        def finish(answer: RAGAnswer) -> RAGAnswer:
            answer.latency_ms = int((time.perf_counter() - started) * 1000)
            return answer

        if not document_ids:
            return finish(RAGAnswer(question=q, answer=NO_DOCS_MESSAGES[lang], language=lang, mode="no_documents"))

        try:
            retrieved = self.retriever.retrieve(q, document_ids, top_k)
        except EmbeddingError as exc:
            return finish(RAGAnswer(question=q, answer=str(exc), language=lang, mode="error"))

        if not retrieved:
            return finish(
                RAGAnswer(
                    question=q,
                    answer=INSUFFICIENT_EVIDENCE_MESSAGES[lang],
                    language=lang,
                    mode="no_results",
                    insufficient_evidence=True,
                )
            )

        sources = build_citations(retrieved)
        if self.generator is None:
            return finish(
                RAGAnswer(
                    question=q,
                    answer=RETRIEVAL_ONLY_MESSAGES[lang],
                    language=lang,
                    mode="retrieval_only",
                    sources=sources,
                )
            )

        try:
            generated = self.generator.generate(q, sources, lang)
        except GenerationError as exc:
            return finish(RAGAnswer(question=q, answer=str(exc), language=lang, mode="error", sources=sources))

        check = validate_citations(generated.answer, sources)
        warnings: list[str] = []
        if check.invalid_ids:
            warnings.append(
                "Removed citation(s) not present in the retrieved evidence: " + ", ".join(check.invalid_ids)
            )
        answer_text = check.text
        if generated.insufficient_evidence:
            answer_text = answer_text or INSUFFICIENT_EVIDENCE_MESSAGES[lang]
        elif not check.valid_ids:
            warnings.append("The answer contains no valid citations. Treat it as unverified.")

        return finish(
            RAGAnswer(
                question=q,
                answer=answer_text,
                language=lang,
                mode="generated",
                sources=sources,
                cited_ids=check.valid_ids,
                invalid_citations=check.invalid_ids,
                insufficient_evidence=generated.insufficient_evidence,
                warnings=warnings,
                usage=generated.usage,
                model=generated.model,
            )
        )
