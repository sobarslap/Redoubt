"""In-process vector index (cosine) over an :class:`Embedder`.

A derived accelerator mapping embedding -> memory_id, matching the storage
division of responsibility: SQLite is authoritative, this is rebuildable. The
brute-force cosine scan is the single-node dev/demo tier; the documented
production swap is sqlite-vec / FAISS / pgvector behind this same surface.
"""

from __future__ import annotations

from aegismem.retrieval.embeddings import Embedder, cosine


class VectorIndex:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._vecs: dict[str, list[float]] = {}

    def add(self, doc_id: str, text: str) -> None:
        self._vecs[doc_id] = self._embedder.embed(text)

    def remove(self, doc_id: str) -> None:
        self._vecs.pop(doc_id, None)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        q = self._embedder.embed(query)
        scored = [(doc_id, cosine(q, v)) for doc_id, v in self._vecs.items()]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k] if top_k is not None else scored

    def __len__(self) -> int:
        return len(self._vecs)
