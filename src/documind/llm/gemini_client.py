"""Grounded answer generation with the Google GenAI SDK.

Used for two providers:
* ``gemini``: Google's Gemini API directly (``GEMINI_API_KEY``).
* ``shopaikey-gemini``: ShopAIKey's Gemini-compatible endpoint, configured with
  ``http_options.base_url`` as documented at https://shopaikey.com/docs/google-genai.
  The ShopAIKey key is only ever sent to the gateway base URL.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from documind.llm.base import (
    CHECK_MAX_TOKENS,
    CHECK_PROMPT,
    AnswerGenerator,
    GeneratedAnswer,
    Turn,
    prepare_history,
)
from documind.llm.errors import GenerationError, classify_error, should_retry
from documind.llm.prompts import build_system_instruction, build_user_prompt
from documind.llm.structured import parse_answer_payload
from documind.models import Citation, TokenUsage
from documind.utils.retry import retry_call

__all__ = ["AnswerGenerator", "GeminiAnswerGenerator", "GeneratedAnswer"]


class _GeminiOutput(BaseModel):
    answer: str
    insufficient_evidence: bool


class GeminiAnswerGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        provider: str = "gemini",
        label: str = "Google Gemini",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        max_retries: int = 2,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise GenerationError("No API key is configured for this provider.", "not_configured")
        if not model:
            raise GenerationError("No model is selected.", "not_configured")
        self.model = model
        self.provider = provider
        self.label = label
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._service = "ShopAIKey" if provider.startswith("shopaikey") else "Gemini"
        if client is None:
            from google import genai
            from google.genai import types

            options: dict[str, Any] = {"timeout": int(timeout_s * 1000)}
            if base_url:
                options["base_url"] = base_url
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(**options))
        self._client = client

    def _call(self, fn: Any, operation: str) -> Any:
        try:
            return retry_call(fn, max_retries=self.max_retries, should_retry=should_retry)
        except Exception as exc:
            state, message = classify_error(exc, operation, self._service)
            raise GenerationError(message, state) from exc

    def check(self) -> str:
        """Verify key + model access.

        Direct Gemini uses the free ``models.get``. Through ShopAIKey a listed
        model is not proof of access for the key's group, so a minimal real
        generation is made instead.
        """
        if self.provider == "gemini":
            info = self._call(lambda: self._client.models.get(model=self.model), "connection check")
            return getattr(info, "name", None) or self.model
        from google.genai import types

        config = types.GenerateContentConfig(max_output_tokens=CHECK_MAX_TOKENS)
        response = self._call(
            lambda: self._client.models.generate_content(model=self.model, contents=CHECK_PROMPT, config=config),
            "connection check",
        )
        return getattr(response, "model_version", None) or self.model

    def generate(
        self,
        question: str,
        sources: Sequence[Citation],
        language: str,
        history: Sequence[Turn] = (),
    ) -> GeneratedAnswer:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=build_system_instruction(language),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=_GeminiOutput,
        )
        contents: list[dict[str, Any]] = [
            {"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]}
            for t in prepare_history(history)
        ]
        contents.append({"role": "user", "parts": [{"text": build_user_prompt(question, sources)}]})
        response = self._call(
            lambda: self._client.models.generate_content(model=self.model, contents=contents, config=config),
            "generation",
        )
        answer, insufficient = _parse_output(response, self._service)
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
            answer=answer,
            insufficient_evidence=insufficient,
            usage=usage,
            model=getattr(response, "model_version", None) or self.model,
        )


def _parse_output(response: Any, service: str = "Gemini") -> tuple[str, bool]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, _GeminiOutput):
        return parsed.answer.strip(), parsed.insufficient_evidence
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise GenerationError(
            f"{service} returned an empty response (it may have been blocked by safety filters).", "error"
        )
    return parse_answer_payload(text)
