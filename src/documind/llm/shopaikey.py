"""ShopAIKey gateway: URL safety, per-protocol base URLs, model discovery, factory.

Documented endpoints (https://shopaikey.com/docs/quickstart):

=========== ============================== ==========================
Protocol    SDK base URL                   Model catalog
=========== ============================== ==========================
openai      https://api.shopaikey.com/v1   GET /v1/models
anthropic   https://api.shopaikey.com      (none; gateway /v1/models)
gemini      https://api.shopaikey.com      GET /v1beta/models
=========== ============================== ==========================

A listed model is *not* proof that a key can use it: each key belongs to a
model group, and calls outside it fail with "No available channel for model".
Only a real request (Settings > Test connection) proves access.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from documind.config import DEFAULT_SHOPAIKEY_ROOT
from documind.llm.errors import GenerationError, classify_error, should_retry
from documind.utils.retry import retry_call

PROTOCOLS: dict[str, str] = {
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic-compatible",
    "gemini": "Gemini-compatible",
}

# A ShopAIKey key must never be sent to these vendors' own endpoints.
_OFFICIAL_HOSTS = {
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
}


def gateway_root(url: str | None) -> str:
    """Normalise a configured base URL to the gateway root (no trailing ``/v1``).

    Accepts ``https://api.shopaikey.com`` or ``https://api.shopaikey.com/v1``.
    Rejects non-HTTPS URLs and the official vendor hosts.
    """
    raw = (url or DEFAULT_SHOPAIKEY_ROOT).strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.hostname:
        raise GenerationError("SHOPAIKEY_BASE_URL must be an https:// URL.", "not_configured")
    host = parsed.hostname.lower()
    if host in _OFFICIAL_HOSTS or host.endswith(".googleapis.com"):
        raise GenerationError(
            "SHOPAIKEY_BASE_URL points at an official vendor API; a ShopAIKey key must only go to the gateway.",
            "not_configured",
        )
    return raw


def base_url_for(protocol: str, root: str) -> str:
    return f"{root}/v1" if protocol == "openai" else root


def build_generator(
    protocol: str,
    *,
    api_key: str,
    model: str,
    root: str,
    max_output_tokens: int,
    timeout_s: float,
    max_retries: int,
) -> Any:
    """Create the adapter for ``protocol``. Raises GenerationError if misconfigured."""
    if protocol not in PROTOCOLS:
        raise GenerationError(f"Unknown ShopAIKey protocol '{protocol}'.", "not_configured")
    url = base_url_for(protocol, gateway_root(root))
    common = {"max_output_tokens": max_output_tokens, "timeout_s": timeout_s, "max_retries": max_retries}
    if protocol == "openai":
        from documind.llm.openai_compat import OpenAICompatibleGenerator

        return OpenAICompatibleGenerator(api_key, model, base_url=url, **common)
    if protocol == "anthropic":
        from documind.llm.anthropic_compat import AnthropicCompatibleGenerator

        return AnthropicCompatibleGenerator(api_key, model, base_url=url, **common)
    from documind.llm.gemini_client import GeminiAnswerGenerator

    return GeminiAnswerGenerator(
        api_key,
        model,
        base_url=url,
        provider="shopaikey-gemini",
        label="ShopAIKey (Gemini-compatible)",
        **common,
    )


def list_models(protocol: str, *, api_key: str, root: str, timeout_s: float = 20.0, client: Any = None) -> list[str]:
    """Model IDs from the gateway's documented catalog for ``protocol``."""
    if not api_key and client is None:
        raise GenerationError("SHOPAIKEY_API_KEY is not configured.", "not_configured")
    root = gateway_root(root)
    try:
        if protocol == "gemini":
            if client is None:
                from google import genai
                from google.genai import types

                client = genai.Client(
                    api_key=api_key, http_options=types.HttpOptions(base_url=root, timeout=int(timeout_s * 1000))
                )
            models = retry_call(lambda: list(client.models.list()), max_retries=1, should_retry=should_retry)
            return sorted({(getattr(m, "name", "") or "").removeprefix("models/") for m in models} - {""})
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=f"{root}/v1", timeout=timeout_s, max_retries=0)
        page = retry_call(lambda: client.models.list(), max_retries=1, should_retry=should_retry)
        ids = sorted({m.id for m in getattr(page, "data", page) if getattr(m, "id", None)})
    except GenerationError:
        raise
    except Exception as exc:
        state, message = classify_error(exc, "model listing", "ShopAIKey")
        raise GenerationError(message, state) from exc
    if protocol == "anthropic":
        # No Anthropic-format catalog is documented; Claude models from the gateway catalog.
        ids = [i for i in ids if "claude" in i.lower()]
    return ids
