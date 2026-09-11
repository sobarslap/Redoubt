"""Retrieval quality metrics — Recall@K, Precision@K, MRR, NDCG.

Pure functions over ranked id lists and a relevant-id set, so the benchmark
suite and CI gates compute quality the same way everywhere. The Phase 3 gate is
explicit: never report a latency number without a quality number beside it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in relevant)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain NDCG@K."""
    dcg = 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(rankings: Sequence[Sequence[str]], relevants: Sequence[set[str]]) -> float:
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants, strict=True)) / len(
        rankings
    )
