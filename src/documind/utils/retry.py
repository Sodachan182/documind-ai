"""Bounded exponential-backoff retry for external API calls."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def status_code_of(exc: BaseException) -> int | None:
    """HTTP status of an SDK error.

    openai/anthropic expose it as ``status_code`` and use ``code`` for a *string*
    error code (e.g. "rate_limit_exceeded"); google-genai uses an integer ``code``.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    code = status_code_of(exc)
    if code is not None:
        return code in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return "timeout" in name or "connect" in name


def retry_call[T](
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    should_retry: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random() / 2)
            sleep(delay)
            attempt += 1
