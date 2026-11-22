"""Embedding providers. Gemini is exercised through a fake SDK client (no network)."""

import math
from types import SimpleNamespace

import pytest

from documind.embeddings.base import EmbeddingError
from documind.embeddings.gemini_embeddings import GeminiEmbedder
from documind.embeddings.local_embeddings import HashingEmbedder


class FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, dim: int, fail_codes: list[int] | None = None) -> None:
        self.dim = dim
        self.fail_codes = list(fail_codes or [])
        self.calls: list[dict] = []

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self.fail_codes:
            raise FakeAPIError(self.fail_codes.pop(0))
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * (self.dim - 2)) for _ in contents]
        )


def _embedder(models: FakeModels, **kw) -> GeminiEmbedder:
    return GeminiEmbedder(
        api_key="", model="gemini-embedding-001", dimension=models.dim, client=SimpleNamespace(models=models), **kw
    )


def test_gemini_embedder_batches_normalises_and_sets_task_type(monkeypatch) -> None:
    models = FakeModels(dim=8)
    emb = _embedder(models, batch_size=2)
    vectors = emb.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3 and len(models.calls) == 2
    assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0)
    assert vectors[0][:2] == pytest.approx([0.6, 0.8])
    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.calls[0]["config"].output_dimensionality == 8
    emb.embed_query("q")
    assert models.calls[-1]["config"].task_type == "RETRIEVAL_QUERY"
    assert emb.key == "gemini:gemini-embedding-001:8"


def test_gemini_embedder_retries_rate_limits(monkeypatch) -> None:
    monkeypatch.setattr("documind.utils.retry.time.sleep", lambda s: None)
    models = FakeModels(dim=4, fail_codes=[429, 503])
    assert len(_embedder(models, max_retries=3).embed_query("q")) == 4
    assert len(models.calls) == 3


def test_gemini_embedder_gives_up_with_safe_message(monkeypatch) -> None:
    monkeypatch.setattr("documind.utils.retry.time.sleep", lambda s: None)
    models = FakeModels(dim=4, fail_codes=[429] * 5)
    with pytest.raises(EmbeddingError, match="rate limit"):
        _embedder(models, max_retries=2).embed_query("q")


def test_invalid_key_is_not_retried() -> None:
    models = FakeModels(dim=4, fail_codes=[403])
    with pytest.raises(EmbeddingError, match="API key is invalid"):
        _embedder(models).embed_query("q")
    assert len(models.calls) == 1


def test_dimension_mismatch_is_rejected() -> None:
    models = FakeModels(dim=4)
    emb = GeminiEmbedder(api_key="", model="m", dimension=8, client=SimpleNamespace(models=models))
    with pytest.raises(EmbeddingError, match="Expected 8-dim"):
        emb.embed_query("q")


def test_missing_api_key_raises() -> None:
    with pytest.raises(EmbeddingError, match="not configured"):
        GeminiEmbedder(api_key="", model="m")


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dimension=256)
    a, b = emb.embed_documents(["annual leave policy", "annual leave policy"])
    assert a == b and len(a) == 256
    assert math.isclose(sum(v * v for v in a), 1.0, rel_tol=1e-9)
    near = emb.embed_query("annual leave days")
    far = emb.embed_query("kubernetes postgres redis")
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, near) > dot(a, far)


def test_gemini_embedder_check() -> None:
    class Models(FakeModels):
        def get(self, *, model):
            raise FakeAPIError(404)

    emb = _embedder(Models(dim=4))
    with pytest.raises(EmbeddingError, match="model was not found"):
        emb.check()
