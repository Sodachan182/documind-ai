"""Logging with secret redaction."""

from __future__ import annotations

import logging
import re

# Google keys start with "AIza"; ShopAIKey / OpenAI-style keys start with "sk-".
_TOKEN_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bsk-[0-9A-Za-z_\-]{16,}"),
]
# Values after key-like labels: api_key=..., Bearer ..., ?key=..., x-api-key: ...
_PAIR_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;&\"']+)"),
    re.compile(r"(?i)(bearer\s+)([^\s,;\"']+)"),
    re.compile(r"(?i)([?&]key=)([^\s&\"']+)"),
    re.compile(r"(?i)(x-(?:api|goog-api)-key[\"']?\s*[:=]\s*[\"']?)([^\s,;\"']+)"),
]


def redact(text: str) -> str:
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    for pattern in _PAIR_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger
