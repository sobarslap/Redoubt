"""Retrieval model factory — real local models with a deterministic fallback.

Phase 3 shipped a `HashingEmbedder` and `OverlapReranker` as free, deterministic
dev stand-ins so retrieval was testable at zero cost. Production wants the real
local models from the `retrieval` dependency group: `bge-small-en-v1.5`
embeddings and a `bge-reranker-base` cross-encoder. Both adapters are
import-guarded; the factory returns the real model when installed and the dev
stand-in otherwise, so the same code runs in keyless CI and in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegismem.retrieval.embeddings import Embedder, HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker, Reranker

# bge-small-en-v1.5 embedding dimensionality (matches PgVectorMemoryStore default).
BGE_SMALL_DIM = 384


@dataclass
class SentenceTransformerEmbedder:
    """`sentence-transformers` adapter. Loads the model lazily on first embed."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    dim: int = BGE_SMALL_DIM
    _model: object | None = field(default=None, repr=False)

    def _ensure(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._ensure()
        vec = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [float(x) for x in vec]


@dataclass
class CrossEncoderReranker:
    """`bge-reranker-base` cross-encoder adapter (lazy load)."""

    model_name: str = "BAAI/bge-reranker-base"
    _model: object | None = field(default=None, repr=False)

    def _ensure(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, document: str) -> float:
        model = self._ensure()
        return float(model.predict([(query, document)])[0])  # type: ignore[attr-defined]


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


def build_embedder(*, prefer_real: bool = True) -> Embedder:
    """Real bge-small embedder if `sentence-transformers` is installed, else the
    deterministic hashing stand-in."""
    if prefer_real and _sentence_transformers_available():
        return SentenceTransformerEmbedder()
    return HashingEmbedder(dim=BGE_SMALL_DIM)


def build_reranker(*, prefer_real: bool = True) -> Reranker:
    if prefer_real and _sentence_transformers_available():
        return CrossEncoderReranker()
    return OverlapReranker()
