"""Retrieval-latency benchmark — quality never travels without a latency number.

Reuses the labelled retrieval corpus and reports P50/P95 over repeated queries.
Kept next to the retrieval quality bench so latency claims are always paired with
Recall/MRR (the Phase 3 gate rule).
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.common import p50_p95
from benchmarks.retrieval.run import CORPUS, QUERIES, build_router


@dataclass(frozen=True)
class LatencyPoint:
    queries: int
    repeats: int
    p50_ms: float
    p95_ms: float


def measure(repeats: int = 20) -> LatencyPoint:
    router = build_router()
    latencies: list[float] = []
    for _ in range(repeats):
        for query in QUERIES:
            latencies.append(router.retrieve(query, documents=CORPUS).latency_ms)
    p50, p95 = p50_p95(latencies)
    return LatencyPoint(len(QUERIES), repeats, round(p50, 4), round(p95, 4))


def main() -> None:
    point = measure()
    print("retrieval latency (hybrid + rerank):")
    print(f"  queries={point.queries} repeats={point.repeats}")
    print(f"  P50 {point.p50_ms:.3f} ms")
    print(f"  P95 {point.p95_ms:.3f} ms")


if __name__ == "__main__":
    main()
