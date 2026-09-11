"""Scale & performance validation (Production Phase P5).

Where ``benchmarks/tokens`` proves the *structural* token delta analytically, this
harness measures the **real** retrieval quality (Recall@K, MRR) and latency
(P50/P95) as the memory store actually grows — the numbers that replace the
extrapolation in ``docs/PERFORMANCE.md`` once run at 100k-1M against pgvector.

It seeds a synthetic-but-realistic corpus: most memories are filler, a labelled
subset are distinctive "target" facts, each with a matching query whose gold
answer is known. The same code runs at 1k in CI (deterministic, fast) and at
100k-1M against a real backend when you point it there.

    python -m benchmarks.scale.run --scales 1000,10000
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.factory import build_embedder
from aegismem.retrieval.metrics import mean_reciprocal_rank, recall_at_k
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex

_SERVICES = ("checkout", "payments", "search", "inventory", "gateway", "auth", "cache", "billing")
_TOPICS = ("latency", "errors", "throughput", "saturation", "timeout", "retry", "backlog", "lag")


def _memory(i: int) -> str:
    svc = _SERVICES[i % len(_SERVICES)]
    topic = _TOPICS[(i // len(_SERVICES)) % len(_TOPICS)]
    return f"incident-{i}: service {svc}-{i} reported {topic} anomaly with code E{i:05d}"


def _target_query(i: int) -> str:
    svc = _SERVICES[i % len(_SERVICES)]
    return f"{svc}-{i} anomaly code E{i:05d}"


@dataclass(frozen=True)
class ScalePoint:
    scale: int
    n_queries: int
    recall_at_k: float
    mrr: float
    p50_ms: float
    p95_ms: float


def measure(
    scale: int, *, n_queries: int = 50, k: int = 5, prefer_real: bool = False
) -> ScalePoint:
    router = JITRouter(
        BM25Index(),
        VectorIndex(build_embedder(prefer_real=prefer_real)),
        OverlapReranker(),
        top_k=k,
    )
    documents: dict[str, str] = {}
    for i in range(scale):
        mid = f"mem_{i}"
        text = _memory(i)
        documents[mid] = text
        router.index(mid, text)

    # Sample target queries spread across the corpus.
    step = max(1, scale // n_queries)
    targets = list(range(0, scale, step))[:n_queries]
    rankings: list[list[str]] = []
    relevants: list[set[str]] = []
    latencies: list[float] = []
    for i in targets:
        res = router.retrieve(_target_query(i), documents=documents)
        rankings.append([r.memory_id for r in res.results])
        relevants.append({f"mem_{i}"})
        latencies.append(res.latency_ms)

    recall = statistics.mean(
        recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)
    )
    mrr = mean_reciprocal_rank(rankings, relevants)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[max(0, int(0.95 * len(latencies)) - 1)]
    return ScalePoint(
        scale, len(targets), round(recall, 4), round(mrr, 4), round(p50, 4), round(p95, 4)
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmarks.scale.run")
    parser.add_argument("--scales", default="1000,10000")
    parser.add_argument("--real", action="store_true", help="use real bge embeddings if installed")
    args = parser.parse_args()
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    print(f"{'scale':>10} {'recall@5':>10} {'mrr':>8} {'p50_ms':>10} {'p95_ms':>10}")
    for scale in scales:
        p = measure(scale, prefer_real=args.real)
        print(
            f"{p.scale:>10} {p.recall_at_k:>10.3f} {p.mrr:>8.3f} "
            f"{p.p50_ms:>10.3f} {p.p95_ms:>10.3f}"
        )


if __name__ == "__main__":
    main()
