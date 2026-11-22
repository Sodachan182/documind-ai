"""Embedding provider interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced. Message is safe to show users."""


class Embedder(ABC):
    provider: str
    model: str
    dimension: int

    @property
    def key(self) -> str:
        """Identifies the vector space. Vectors from different keys are incompatible."""
        return f"{self.provider}:{self.model}:{self.dimension}"

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]
