"""JIT retrieval router — adaptive hybrid retrieval with measurable guarantees.

Not "top15 -> top3 as a vibe". The router:

* picks a **strategy** from the query shape (exact identifier -> BM25-heavy;
  conceptual question -> vector-heavy; ambiguous/high-value -> full hybrid+rerank)
  and logs the choice;
* fuses BM25 + vector scores (min-max normalized) by the strategy weights;
* reranks the top candidates with a cross-encoder (or falls back to hybrid order);
* applies a **relevance threshold** and returns top-K.

**Critical rule:** if nothing clears the threshold, return EMPTY MEMORY rather
than inject a vaguely-related fact. Empty is safer than wrong.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field

from aegismem.retrieval.bm25 import BM25Index, tokenize
from aegismem.retrieval.reranker import Reranker
from aegismem.retrieval.vector import VectorIndex


class Strategy(StrEnum):
    BM25_HEAVY = "bm25_heavy"
    VECTOR_HEAVY = "vector_heavy"
    FULL_HYBRID = "full_hybrid"


# (bm25_weight, vector_weight) per strategy.
_WEIGHTS: dict[Strategy, tuple[float, float]] = {
    Strategy.BM25_HEAVY: (0.8, 0.2),
    Strategy.VECTOR_HEAVY: (0.2, 0.8),
    Strategy.FULL_HYBRID: (0.5, 0.5),
}

_IDENTIFIER = re.compile(
    r"""(
        \b\d{1,3}(?:\.\d{1,3}){3}\b        # IPv4
      | \b[\w-]+\.[\w.-]+\b                # hostname / domain
      | \b[A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b   # SCREAMING_SNAKE token / key name
      | \b[0-9a-f]{8,}\b                   # long hex id
      | =                                   # key=value form
    )""",
    re.VERBOSE,
)
_QUESTION = re.compile(r"\b(why|how|what|when|where|which|should|explain|cause)\b", re.I)


class Retrieved(BaseModel):
    memory_id: str
    score: float
    bm25: float = 0.0
    vector: float = 0.0
    rerank: float | None = None


class RouterResult(BaseModel):
    query: str
    strategy: Strategy
    empty: bool
    results: list[Retrieved] = Field(default_factory=list)
    candidates_considered: int = 0
    latency_ms: float = 0.0


def choose_strategy(query: str) -> Strategy:
    """Adaptive strategy selection from query shape (logged per run)."""
    has_identifier = bool(_IDENTIFIER.search(query))
    is_question = bool(_QUESTION.search(query)) or len(tokenize(query)) >= 8
    if has_identifier and not is_question:
        return Strategy.BM25_HEAVY
    if is_question and not has_identifier:
        return Strategy.VECTOR_HEAVY
    return Strategy.FULL_HYBRID


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class JITRouter:
    """Hybrid BM25 + vector retrieval with adaptive strategy, rerank, threshold."""

    def __init__(
        self,
        bm25: BM25Index,
        vector: VectorIndex,
        reranker: Reranker | None = None,
        *,
        relevance_threshold: float = 0.35,
        min_vector_sim: float = 0.35,
        top_k: int = 3,
        rerank_top_n: int = 20,
        candidate_pool: int = 50,
    ) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._reranker = reranker
        self._threshold = relevance_threshold
        self._min_sim = min_vector_sim
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._pool = candidate_pool

    def index(self, doc_id: str, text: str) -> None:
        self._bm25.add(doc_id, text)
        self._vector.add(doc_id, text)

    def remove(self, doc_id: str) -> None:
        self._bm25.remove(doc_id)
        self._vector.remove(doc_id)

    def retrieve(self, query: str, *, documents: dict[str, str] | None = None) -> RouterResult:
        """Retrieve for ``query``. ``documents`` (id -> content) is needed only
        when a reranker is configured, to score candidate text."""
        start = time.perf_counter()
        strategy = choose_strategy(query)
        w_bm25, w_vec = _WEIGHTS[strategy]

        bm25_raw = dict(self._bm25.search(query, top_k=self._pool))
        # Vector scores are cosine over non-negative embeddings, so already an
        # absolute [0, 1] signal; BM25 is unbounded, so it is min-max scaled for
        # fusion ordering only.
        vec_raw = dict(self._vector.search(query, top_k=self._pool))
        bm25_n = _minmax(bm25_raw)

        candidate_ids = set(bm25_raw) | set(vec_raw)
        # Eligibility gate protects the EMPTY MEMORY guarantee: a candidate counts
        # only on an *absolute* signal — a real lexical hit, or cosine >= min_sim —
        # so min-max scaling can never promote near-zero similarity to a "match".
        eligible = {
            cid
            for cid in candidate_ids
            if bm25_raw.get(cid, 0.0) > 0.0 or vec_raw.get(cid, 0.0) >= self._min_sim
        }
        fused: dict[str, float] = {
            cid: w_bm25 * bm25_n.get(cid, 0.0) + w_vec * vec_raw.get(cid, 0.0) for cid in eligible
        }
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        rerank_scores: dict[str, float] = {}
        if self._reranker is not None and documents is not None:
            head = ordered[: self._rerank_top_n]
            for cid, _ in head:
                text = documents.get(cid)
                if text is not None:
                    rerank_scores[cid] = self._reranker.score(query, text)
            if rerank_scores:
                rr_n = _minmax(rerank_scores)
                ordered = sorted(
                    ((cid, rr_n.get(cid, fused[cid])) for cid, _ in ordered),
                    key=lambda kv: kv[1],
                    reverse=True,
                )

        results: list[Retrieved] = []
        for cid, score in ordered:
            if score < self._threshold:
                continue
            results.append(
                Retrieved(
                    memory_id=cid,
                    score=round(score, 6),
                    bm25=round(bm25_n.get(cid, 0.0), 6),
                    vector=round(vec_raw.get(cid, 0.0), 6),
                    rerank=round(rerank_scores[cid], 6) if cid in rerank_scores else None,
                )
            )
            if len(results) >= self._top_k:
                break

        latency_ms = (time.perf_counter() - start) * 1000
        return RouterResult(
            query=query,
            strategy=strategy,
            empty=len(results) == 0,  # the EMPTY MEMORY guarantee
            results=results,
            candidates_considered=len(candidate_ids),
            latency_ms=round(latency_ms, 4),
        )


def build_documents(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Convenience: id/content pairs -> the ``documents`` map ``retrieve`` wants."""
    return dict(pairs)
