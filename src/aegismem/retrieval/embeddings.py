"""Embedding provider interface + a dependency-free default.

``Embedder`` is the seam. The production embedder is
``sentence-transformers`` ``bge-small-en-v1.5`` (local, free); it plugs in here
without the router knowing. :class:`HashingEmbedder` is the zero-dependency
fallback — a deterministic hashed-bag-of-words projection — so the retrieval
stack builds, tests, and benchmarks on any interpreter (notably Python 3.14,
where the ML wheels may lag) without a heavy install.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from aegismem.retrieval.bm25 import tokenize


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic hashed term-frequency vectors with L2 normalization.

    Tokens are hashed into ``dim`` buckets and weighted by sqrt term frequency;
    the result is L2-normalized so dot product is cosine similarity. Overlapping
    vocabulary yields high similarity — enough for meaningful hybrid ranking and
    reproducible metrics without any model download.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vec
        counts: dict[int, float] = {}
        for tok in tokens:
            counts[self._bucket(tok)] = counts.get(self._bucket(tok), 0.0) + 1.0
        for bucket, freq in counts.items():
            vec[bucket] = math.sqrt(freq)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; inputs from :class:`HashingEmbedder` are already unit-norm."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
