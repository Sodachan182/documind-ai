"""Offline lexical embedder (feature hashing).

This is NOT a neural embedding model. It maps unigrams and bigrams into a
fixed-size vector with signed feature hashing and sublinear TF weighting. It
exists so that the full ingestion -> ChromaDB -> retrieval path can run (and be
evaluated) without an API key, e.g. in CI. It only matches shared words, so it
does not handle paraphrases or cross-lingual queries. The UI and evaluation
reports always label it as "offline lexical".
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from documind.embeddings.base import Embedder, l2_normalize

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has have how in is it its of on or that the this "
        "to was were what when where which who will with do does can our your their per"
    ).split()
)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return [t for t in _TOKEN.findall(text) if t not in _STOPWORDS]


class HashingEmbedder(Embedder):
    provider = "local"
    model = "hashing-lexical-v1"

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dimension, (1.0 if (value >> 63) & 1 else -1.0)

    def _embed(self, text: str) -> list[float]:
        tokens = _tokens(text)
        features = Counter(tokens)
        features.update(f"{a}_{b}" for a, b in itertools.pairwise(tokens))
        vector = [0.0] * self.dimension
        for feature, count in features.items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1.0 + math.log(count))
        return l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
