"""Optional ShopAIKey embeddings through the OpenAI-format ``/v1/embeddings`` endpoint.

Opt-in only (``DOCUMIND_EMBEDDING_PROVIDER=shopaikey``). The model and its output
dimension must be configured explicitly (``DOCUMIND_EMBEDDING_MODEL``,
``DOCUMIND_EMBEDDING_DIMENSION``); every returned vector is checked against that
dimension, so a mismatch fails loudly instead of mixing vector spaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from documind.embeddings.base import Embedder, EmbeddingError, l2_normalize
from documind.llm.errors import GenerationError, classify_error, should_retry
from documind.utils.retry import retry_call


class ShopAIKeyEmbedder(Embedder):
    provider = "shopaikey"

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        *,
        base_url: str,
        batch_size: int = 32,
        max_retries: int = 2,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise EmbeddingError("SHOPAIKEY_API_KEY is required for ShopAIKey embeddings.")
        if not model:
            raise EmbeddingError("DOCUMIND_EMBEDDING_MODEL is required for ShopAIKey embeddings.")
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self._client = client

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = retry_call(
                    lambda b=batch: self._client.embeddings.create(model=self.model, input=b),
                    max_retries=self.max_retries,
                    should_retry=should_retry,
                )
            except Exception as exc:
                raise EmbeddingError(classify_error(exc, "embedding", "ShopAIKey")[1]) from exc
            data = sorted(getattr(response, "data", None) or [], key=lambda d: getattr(d, "index", 0))
            if len(data) != len(batch):
                raise EmbeddingError(f"Embedding API returned {len(data)} vectors for {len(batch)} inputs.")
            for item in data:
                values = list(item.embedding or [])
                if len(values) != self.dimension:
                    raise EmbeddingError(
                        f"{self.model} returned {len(values)}-dim vectors; DOCUMIND_EMBEDDING_DIMENSION is "
                        f"{self.dimension}. Set it to {len(values)} and re-index."
                    )
                vectors.append(l2_normalize(values))
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def check(self) -> str:
        """One tiny real embedding request (verifies key, model access and dimension)."""
        try:
            self._embed(["ping"])
        except EmbeddingError as exc:
            raise GenerationError(str(exc), "error") from exc
        return self.model
