"""Hand-rolled Okapi BM25 lexical index.

Hand-rolled (rather than ``rank-bm25``) on purpose: the whole project pitch is
line-by-line control, and BM25 is the lexical accelerator for the source of
truth, so its scoring is worth owning. The index is a derived, rebuildable
structure — memory ids are the only durable reference.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-process BM25 over ``{doc_id: text}`` with the standard k1/b knobs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        tokens = tokenize(text)
        tf = Counter(tokens)
        self._docs[doc_id] = tf
        self._doc_len[doc_id] = len(tokens)
        for term in tf:
            self._df[term] += 1
        self._recompute_avgdl()

    def remove(self, doc_id: str) -> None:
        tf = self._docs.pop(doc_id, None)
        if tf is None:
            return
        self._doc_len.pop(doc_id, None)
        for term in tf:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._recompute_avgdl()

    def _recompute_avgdl(self) -> None:
        self._avgdl = sum(self._doc_len.values()) / len(self._doc_len) if self._doc_len else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # Okapi idf with +1 to stay non-negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scores: dict[str, float] = {}
        for doc_id, tf in self._docs.items():
            dl = self._doc_len[doc_id]
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k] if top_k is not None else ranked

    def __len__(self) -> int:
        return len(self._docs)
