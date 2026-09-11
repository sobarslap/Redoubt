"""Retrieval benchmark — reports quality *and* latency together.

Run: ``uv run python -m benchmarks.retrieval.run``

Builds a small labelled corpus, runs a query set through the JIT router, and
prints Recall@K, Precision@K, MRR, NDCG@K, empty-result rate, and P50/P95
latency. This is the reproducible harness behind ``docs/RETRIEVAL.md``; the CI
gate blocks a quality regression, so the numbers travel with the latency.
"""

from __future__ import annotations

import statistics

from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex

CORPUS: dict[str, str] = {
    "m_pg": "Primary database migrated to PostgreSQL 16 during the change window",
    "m_mysql": "Legacy checkout service used MySQL 8.0 before the migration",
    "m_redis": "Cache layer runs Redis 7 with 2GB maxmemory and LRU eviction",
    "m_host": "The metrics API is reachable at metrics.prod.internal:9090",
    "m_sop": "Runbook: on connection pool exhaustion restart the service and scale replicas",
    "m_alert": "PagerDuty alert fired for checkout p95 latency above the 800ms SLO",
    "m_deploy": "Deploy cadence is weekly on Thursdays via the blue-green pipeline",
    "m_owner": "The payments service is owned by the transactions platform team",
}

# query -> the gold-relevant memory id
QUERIES: dict[str, str] = {
    "what database did we migrate the checkout service to": "m_pg",
    "redis cache eviction policy and maxmemory": "m_redis",
    "metrics.prod.internal:9090": "m_host",
    "how do we recover from connection pool exhaustion": "m_sop",
    "why is checkout latency breaching the SLO": "m_alert",
    "when do we deploy": "m_deploy",
    "who owns the payments service": "m_owner",
}


def build_router() -> JITRouter:
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    for doc_id, text in CORPUS.items():
        router.index(doc_id, text)
    return router


def main() -> None:
    router = build_router()
    rankings: list[list[str]] = []
    relevants: list[set[str]] = []
    latencies: list[float] = []
    empties = 0

    for query, gold in QUERIES.items():
        res = router.retrieve(query, documents=CORPUS)
        ranked = [r.memory_id for r in res.results]
        rankings.append(ranked)
        relevants.append({gold})
        latencies.append(res.latency_ms)
        empties += int(res.empty)

    k = 3
    pairs = list(zip(rankings, relevants, strict=True))
    recall = statistics.mean(recall_at_k(r, rel, k) for r, rel in pairs)
    precision = statistics.mean(precision_at_k(r, rel, k) for r, rel in pairs)
    ndcg = statistics.mean(ndcg_at_k(r, rel, k) for r, rel in pairs)
    mrr = mean_reciprocal_rank(rankings, relevants)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[max(0, int(0.95 * len(latencies)) - 1)]

    print(f"corpus={len(CORPUS)} queries={len(QUERIES)} k={k}")
    print(f"Recall@{k}    {recall:.3f}")
    print(f"Precision@{k} {precision:.3f}")
    print(f"NDCG@{k}      {ndcg:.3f}")
    print(f"MRR          {mrr:.3f}")
    print(f"empty_rate   {empties / len(QUERIES):.3f}")
    print(f"latency_p50  {p50:.3f} ms")
    print(f"latency_p95  {p95:.3f} ms")


if __name__ == "__main__":
    main()
