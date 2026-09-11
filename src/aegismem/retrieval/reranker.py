"""Cross-encoder rerank interface + a dependency-free default.

Production reranker is a cross-encoder (``bge-reranker-base``); it satisfies
:class:`Reranker` and drops in without router changes. :class:`OverlapReranker`
is the zero-dependency fallback — a token-overlap (Jaccard-ish) scorer — so the
pipeline reranks and benchmarks without a model download. The failure policy
"reranker -> fall back to hybrid score ordering" is honoured by the router when
no reranker is supplied.
"""

from __future__ import annotations

import math
from typing import Protocol

from aegismem.retrieval.bm25 import tokenize


class Reranker(Protocol):
    def score(self, query: str, document: str) -> float: ...


class OverlapReranker:
    """Weighted token-overlap: recall of query terms, damped by document length."""

    def score(self, query: str, document: str) -> float:
        q = set(tokenize(query))
        d = set(tokenize(document))
        if not q or not d:
            return 0.0
        return len(q & d) / len(q) * math.sqrt(len(q & d) / len(q | d))
