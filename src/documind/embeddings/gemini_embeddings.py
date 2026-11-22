"""Gemini text embeddings via the official ``google-genai`` SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from documind.embeddings.base import Embedder, EmbeddingError, l2_normalize
from documind.llm.errors import describe_api_error
from documind.utils.retry import retry_call


class GeminiEmbedder(Embedder):
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int = 768,
        *,
        batch_size: int = 50,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise EmbeddingError("GEMINI_API_KEY is not configured.")
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
            )
        self._client = client

    def check(self) -> str:
        """Verify the key and embedding model without embedding anything (``models.get``)."""
        try:
            info = self._client.models.get(model=self.model)
        except Exception as exc:
            raise EmbeddingError(describe_api_error(exc, "connection check")) from exc
        return getattr(info, "name", None) or self.model

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimension)
            try:
                response = retry_call(
                    lambda b=batch, c=config: self._client.models.embed_content(model=self.model, contents=b, config=c),
                    max_retries=self.max_retries,
                )
            except Exception as exc:
                raise EmbeddingError(describe_api_error(exc, "embedding")) from exc
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise EmbeddingError(f"Embedding API returned {len(embeddings)} vectors for {len(batch)} inputs.")
            for item in embeddings:
                values = list(item.values or [])
                if len(values) != self.dimension:
                    raise EmbeddingError(f"Expected {self.dimension}-dim embeddings, got {len(values)}.")
                # Truncated (non-3072) Gemini embeddings must be re-normalised.
                vectors.append(l2_normalize(values))
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
