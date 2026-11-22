"""Translate provider SDK exceptions into safe, user-facing messages and states.

Works for the google-genai, openai and anthropic SDKs (status code read from
``code`` or ``status_code``). Message matching follows the error strings that
ShopAIKey documents at https://shopaikey.com/docs/troubleshooting.
"""

from __future__ import annotations

from typing import Final

from documind.utils.logging_utils import redact
from documind.utils.retry import RETRYABLE_STATUS, status_code_of

# Connection states shown in Settings.
AUTH_FAILED: Final = "auth_failed"
INSUFFICIENT_BALANCE: Final = "insufficient_balance"
MODEL_UNAVAILABLE: Final = "model_unavailable"
RATE_LIMITED: Final = "rate_limited"
TIMEOUT: Final = "timeout"
BAD_REQUEST: Final = "bad_request"
SERVICE_ERROR: Final = "service_error"
CONNECTION_ERROR: Final = "connection_error"
ERROR: Final = "error"

_QUOTA_HINTS = ("quota is not enough", "quota is exhausted", "insufficient quota", "insufficient balance")
_MODEL_HINTS = (
    "no available channel",
    "no access to model",
    "is not supported",
    "model_not_found",
    "model not found",
    "does not exist",
)


class GenerationError(RuntimeError):
    """Raised when answer generation fails. Message is safe to show users."""

    def __init__(self, message: str, state: str = ERROR) -> None:
        super().__init__(message)
        self.state = state


def _detail(exc: BaseException) -> str:
    parts = [str(getattr(exc, "message", "") or ""), str(exc)]
    body = getattr(exc, "body", None)
    if body:
        parts.append(str(body))
    return redact(" ".join(p for p in parts if p))


def classify_error(exc: BaseException, operation: str = "request", provider: str = "Gemini") -> tuple[str, str]:
    """Return ``(state, safe_message)``; adds the service's own (redacted) reason when it gave one."""
    state, message = _classify(exc, operation, provider)
    if state in (TIMEOUT, CONNECTION_ERROR, BAD_REQUEST, ERROR):
        return state, message
    reason = redact(str(getattr(exc, "message", "") or "")).strip()
    if reason:
        message = f"{message} Service said: {reason[:180]}"
    return state, message


def _classify(exc: BaseException, operation: str, provider: str) -> tuple[str, str]:
    code = status_code_of(exc)
    detail = _detail(exc)
    low = detail.lower()
    name = type(exc).__name__.lower()

    if any(h in low for h in _QUOTA_HINTS):
        return INSUFFICIENT_BALANCE, (
            f"{provider} {operation} failed: insufficient balance or quota for this key "
            "(or the upstream channel is temporarily out of quota)."
        )
    if (code in (404, 503) and any(h in low for h in _MODEL_HINTS)) or code == 404:
        return MODEL_UNAVAILABLE, (
            f"{provider} {operation} failed: the model is not available to this key "
            "(wrong model ID, or not in the key's model group)."
        )
    if any(h in low for h in _MODEL_HINTS) and code in (400, 403):
        return MODEL_UNAVAILABLE, f"{provider} {operation} failed: this key has no access to the selected model."
    if code == 401:
        return AUTH_FAILED, f"{provider} {operation} failed: the API key is invalid or disabled."
    if code == 403:
        return AUTH_FAILED, f"{provider} {operation} failed: the API key is invalid or lacks permission."
    if code == 429:
        return RATE_LIMITED, f"{provider} {operation} failed: rate limit or overloaded upstream. Wait and retry."
    if code == 400:
        short = detail[:200]
        return BAD_REQUEST, f"{provider} {operation} failed: invalid request. {short}".strip()
    if code is not None and code >= 500:
        return SERVICE_ERROR, f"{provider} {operation} failed: the service is temporarily unavailable ({code})."
    if "timeout" in name:
        return TIMEOUT, f"{provider} {operation} timed out. Try again or increase DOCUMIND_REQUEST_TIMEOUT_S."
    if "connection" in name or "connect" in name:
        return CONNECTION_ERROR, f"{provider} {operation} failed: could not connect to the service."
    return ERROR, f"{provider} {operation} failed ({type(exc).__name__})."


def describe_api_error(exc: BaseException, operation: str, provider: str = "Gemini") -> str:
    return classify_error(exc, operation, provider)[1]


def should_retry(exc: BaseException) -> bool:
    """Retry only transient failures; never model-access or balance errors."""
    state, _ = classify_error(exc)
    if state in (MODEL_UNAVAILABLE, INSUFFICIENT_BALANCE, AUTH_FAILED, BAD_REQUEST):
        return False
    code = status_code_of(exc)
    if code is not None:
        return code in RETRYABLE_STATUS
    return state in (TIMEOUT, CONNECTION_ERROR)
