"""Translate Gemini SDK exceptions into safe, user-facing messages."""

from __future__ import annotations

from documind.utils.logging_utils import redact
from documind.utils.retry import status_code_of


class GenerationError(RuntimeError):
    """Raised when answer generation fails. Message is safe to show users."""


def describe_api_error(exc: BaseException, operation: str) -> str:
    code = status_code_of(exc)
    if code in (401, 403):
        return f"Gemini {operation} failed: the API key is invalid or lacks permission."
    if code == 404:
        return f"Gemini {operation} failed: the configured model was not found. Check the model name in .env."
    if code == 429:
        return f"Gemini {operation} failed: rate limit or quota exceeded. Wait a moment and retry."
    if code == 400:
        detail = redact(str(getattr(exc, "message", "") or ""))[:200]
        return f"Gemini {operation} failed: invalid request. {detail}".strip()
    if code is not None and code >= 500:
        return f"Gemini {operation} failed: the service is temporarily unavailable ({code})."
    if "timeout" in type(exc).__name__.lower():
        return f"Gemini {operation} timed out. Try again or increase GEMINI_TIMEOUT_S."
    return f"Gemini {operation} failed ({type(exc).__name__})."
