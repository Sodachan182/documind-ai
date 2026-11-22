"""Parse the model's JSON answer payload, tolerating common wrappers.

Providers reached through an OpenAI- or Anthropic-compatible gateway do not all
support enforced JSON output, so the payload is requested in the prompt and
parsed leniently here: bare JSON, JSON inside a ```json fence, or the first
``{...}`` object in the text. If nothing parses, the raw text is used as the
answer; citations are still validated downstream.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _try(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("answer"), str) else None


def parse_answer_payload(text: str) -> tuple[str, bool]:
    """Return ``(answer, insufficient_evidence)``."""
    text = (text or "").strip()
    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        payload = _try(candidate)
        if payload is not None:
            return payload["answer"].strip(), bool(payload.get("insufficient_evidence", False))
    return text, False
