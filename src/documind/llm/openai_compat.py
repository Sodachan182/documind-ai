"""ShopAIKey OpenAI-compatible adapter (official ``openai`` SDK, Chat Completions).

Base URL ``https://api.shopaikey.com/v1`` (https://shopaikey.com/docs/openai-format).
JSON output is requested in the prompt rather than via ``response_format``,
because not every model behind the gateway accepts that parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from documind.llm.base import CHECK_MAX_TOKENS, CHECK_PROMPT, GeneratedAnswer, Turn, prepare_history
from documind.llm.errors import BAD_REQUEST, GenerationError, classify_error, should_retry
from documind.llm.prompts import build_system_instruction, build_user_prompt
from documind.llm.structured import parse_answer_payload
from documind.models import Citation, TokenUsage
from documind.utils.retry import retry_call, status_code_of

MAX_PARAM_ADJUSTMENTS = 2


def _adjust_params(exc: BaseException, params: dict[str, Any]) -> bool:
    """Adapt to models that reject a parameter (e.g. newer reasoning models).

    Returns True if ``params`` was changed and the request may be retried once.
    """
    if status_code_of(exc) != 400:
        return False
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    if "max_tokens" in text and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
        return True
    if "max_completion_tokens" in text and "max_completion_tokens" in params:
        params["max_tokens"] = params.pop("max_completion_tokens")
        return True
    if "temperature" in text and "temperature" in params:
        params.pop("temperature")
        return True
    return False


class OpenAICompatibleGenerator:
    provider = "shopaikey-openai"
    label = "ShopAIKey (OpenAI-compatible)"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        max_retries: int = 2,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise GenerationError("SHOPAIKEY_API_KEY is not configured.", "not_configured")
        if not model:
            raise GenerationError("No ShopAIKey model is selected.", "not_configured")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        if client is None:
            from openai import OpenAI

            # SDK-level retries are disabled; retries are bounded by retry_call below.
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self._client = client

    def _create(self, params: dict[str, Any], operation: str) -> Any:
        adjustments = 0
        while True:
            try:
                return retry_call(
                    lambda: self._client.chat.completions.create(**params),
                    max_retries=self.max_retries,
                    should_retry=should_retry,
                )
            except Exception as exc:
                if adjustments < MAX_PARAM_ADJUSTMENTS and _adjust_params(exc, params):
                    adjustments += 1
                    continue
                state, message = classify_error(exc, operation, "ShopAIKey")
                raise GenerationError(message, state) from exc

    def check(self) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": CHECK_PROMPT}],
            "max_tokens": CHECK_MAX_TOKENS,
        }
        response = self._create(params, "connection check")
        return getattr(response, "model", None) or self.model

    def generate(
        self,
        question: str,
        sources: Sequence[Citation],
        language: str,
        history: Sequence[Turn] = (),
    ) -> GeneratedAnswer:
        messages: list[dict[str, str]] = [{"role": "system", "content": build_system_instruction(language)}]
        messages += [{"role": t["role"], "content": t["content"]} for t in prepare_history(history)]
        messages.append({"role": "user", "content": build_user_prompt(question, sources)})
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        response = self._create(params, "generation")
        choices = getattr(response, "choices", None) or []
        text = (choices[0].message.content or "") if choices else ""
        if not text.strip():
            raise GenerationError("ShopAIKey returned an empty response.", BAD_REQUEST)
        answer, insufficient = parse_answer_payload(text)
        usage = getattr(response, "usage", None)
        return GeneratedAnswer(
            answer=answer,
            insufficient_evidence=insufficient,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
            if usage is not None
            else None,
            model=getattr(response, "model", None) or self.model,
        )

    def list_models(self) -> list[str]:
        try:
            page = retry_call(lambda: self._client.models.list(), max_retries=1, should_retry=should_retry)
        except Exception as exc:
            state, message = classify_error(exc, "model listing", "ShopAIKey")
            raise GenerationError(message, state) from exc
        return sorted({m.id for m in getattr(page, "data", page) if getattr(m, "id", None)})
