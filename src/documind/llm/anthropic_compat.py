"""ShopAIKey Anthropic-compatible adapter (official ``anthropic`` SDK, Messages API).

Base URL ``https://api.shopaikey.com`` with no ``/v1`` suffix; the SDK adds
``/v1/messages`` itself (https://shopaikey.com/docs/anthropic-format).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from documind.llm.base import CHECK_MAX_TOKENS, CHECK_PROMPT, GeneratedAnswer, Turn, prepare_history
from documind.llm.errors import BAD_REQUEST, GenerationError, classify_error, should_retry
from documind.llm.prompts import build_system_instruction, build_user_prompt
from documind.llm.structured import parse_answer_payload
from documind.models import Citation, TokenUsage
from documind.utils.retry import retry_call


class AnthropicCompatibleGenerator:
    provider = "shopaikey-anthropic"
    label = "ShopAIKey (Anthropic-compatible)"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str,
        max_output_tokens: int = 1024,
        max_retries: int = 2,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise GenerationError("SHOPAIKEY_API_KEY is not configured.", "not_configured")
        if not model:
            raise GenerationError("No ShopAIKey model is selected.", "not_configured")
        if base_url.rstrip("/").endswith("/v1"):
            raise GenerationError("The Anthropic-compatible base URL must not end with /v1.", "not_configured")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        if client is None:
            from anthropic import Anthropic

            # Explicit key and base URL: the SDK never falls back to ANTHROPIC_* env vars.
            client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self._client = client

    def _create(self, operation: str, **params: Any) -> Any:
        try:
            return retry_call(
                lambda: self._client.messages.create(**params),
                max_retries=self.max_retries,
                should_retry=should_retry,
            )
        except Exception as exc:
            state, message = classify_error(exc, operation, "ShopAIKey")
            raise GenerationError(message, state) from exc

    def check(self) -> str:
        response = self._create(
            "connection check",
            model=self.model,
            max_tokens=CHECK_MAX_TOKENS,
            messages=[{"role": "user", "content": CHECK_PROMPT}],
        )
        return getattr(response, "model", None) or self.model

    def generate(
        self,
        question: str,
        sources: Sequence[Citation],
        language: str,
        history: Sequence[Turn] = (),
    ) -> GeneratedAnswer:
        messages = [{"role": t["role"], "content": t["content"]} for t in prepare_history(history)]
        messages.append({"role": "user", "content": build_user_prompt(question, sources)})
        response = self._create(
            "generation",
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=build_system_instruction(language),
            messages=messages,
        )
        text = "".join(
            getattr(block, "text", "")
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", "text") == "text"
        )
        if not text.strip():
            raise GenerationError("ShopAIKey returned an empty response.", BAD_REQUEST)
        answer, insufficient = parse_answer_payload(text)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total = prompt_tokens + output_tokens if prompt_tokens is not None and output_tokens is not None else None
        return GeneratedAnswer(
            answer=answer,
            insufficient_evidence=insufficient,
            usage=TokenUsage(prompt_tokens=prompt_tokens, output_tokens=output_tokens, total_tokens=total)
            if usage is not None
            else None,
            model=getattr(response, "model", None) or self.model,
        )
