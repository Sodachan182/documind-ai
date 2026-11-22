"""Grounded answer generation with Gemini (structured JSON output)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from documind.llm.errors import GenerationError, describe_api_error
from documind.llm.prompts import build_system_instruction, build_user_prompt
from documind.models import Citation, TokenUsage
from documind.utils.retry import retry_call


class GeneratedAnswer(BaseModel):
    answer: str
    insufficient_evidence: bool = False
    usage: TokenUsage | None = None
    model: str | None = None


class AnswerGenerator(Protocol):
    model: str

    def generate(self, question: str, sources: Sequence[Citation], language: str) -> GeneratedAnswer: ...


class _GeminiOutput(BaseModel):
    answer: str
    insufficient_evidence: bool


class GeminiAnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise GenerationError("GEMINI_API_KEY is not configured.")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
            )
        self._client = client

    def check(self) -> str:
        """Verify the key and model without generating tokens (``models.get``)."""
        try:
            info = self._client.models.get(model=self.model)
        except Exception as exc:
            raise GenerationError(describe_api_error(exc, "connection check")) from exc
        return getattr(info, "name", None) or self.model

    def generate(self, question: str, sources: Sequence[Citation], language: str) -> GeneratedAnswer:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=build_system_instruction(language),
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=_GeminiOutput,
        )
        prompt = build_user_prompt(question, sources)
        try:
            response = retry_call(
                lambda: self._client.models.generate_content(model=self.model, contents=prompt, config=config),
                max_retries=self.max_retries,
            )
        except Exception as exc:
            raise GenerationError(describe_api_error(exc, "generation")) from exc

        parsed = _parse_output(response)
        meta = getattr(response, "usage_metadata", None)
        usage = (
            TokenUsage(
                prompt_tokens=getattr(meta, "prompt_token_count", None),
                output_tokens=getattr(meta, "candidates_token_count", None),
                total_tokens=getattr(meta, "total_token_count", None),
            )
            if meta is not None
            else None
        )
        return GeneratedAnswer(
            answer=parsed.answer.strip(),
            insufficient_evidence=parsed.insufficient_evidence,
            usage=usage,
            model=getattr(response, "model_version", None) or self.model,
        )


def _parse_output(response: Any) -> _GeminiOutput:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, _GeminiOutput):
        return parsed
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise GenerationError("Gemini returned an empty response (it may have been blocked by safety filters).")
    try:
        return _GeminiOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        # Model ignored the JSON format: keep the text, citations are still validated.
        return _GeminiOutput(answer=text, insufficient_evidence=False)
