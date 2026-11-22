"""Common interface for answer generators (Gemini direct and the ShopAIKey adapters)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel

from documind.models import Citation, TokenUsage

CHECK_PROMPT = "Reply with the single word OK."
CHECK_MAX_TOKENS = 16
MAX_HISTORY_TURNS = 4  # previous messages passed to the model (2 exchanges)
MAX_HISTORY_CHARS = 1500  # per message

_MARKERS = re.compile(r"\s*\[S\d+\]")


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GeneratedAnswer(BaseModel):
    answer: str
    insufficient_evidence: bool = False
    usage: TokenUsage | None = None
    model: str | None = None


class AnswerGenerator(Protocol):
    """What the RAG service needs from any provider."""

    model: str
    provider: str  # stable id, e.g. "gemini", "shopaikey-openai"
    label: str  # human label, e.g. "ShopAIKey (OpenAI-compatible)"

    def generate(
        self,
        question: str,
        sources: Sequence[Citation],
        language: str,
        history: Sequence[Turn] = (),
    ) -> GeneratedAnswer: ...

    def check(self) -> str:
        """Make one real request; return the model id reported by the service."""
        ...


def prepare_history(history: Sequence[Turn]) -> list[Turn]:
    """Last few turns, citation markers removed, length-capped.

    Earlier answers cite *earlier* sources, so their [S#] markers would be
    ambiguous next to the current source block; they are stripped.
    """
    cleaned: list[Turn] = []
    for turn in list(history)[-MAX_HISTORY_TURNS:]:
        text = _MARKERS.sub("", turn["content"]).strip()[:MAX_HISTORY_CHARS]
        if text:
            cleaned.append(Turn(role=turn["role"], content=text))
    while cleaned and cleaned[0]["role"] != "user":  # providers expect a user turn first
        cleaned.pop(0)
    return cleaned
