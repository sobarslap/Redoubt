"""Phase 3 gate: JIT router — BM25, hybrid, rerank, threshold, adaptive strategy.

Quality metrics are asserted alongside behaviour, per the phase gate: never a
latency number without a retrieval-quality number beside it.
"""

from __future__ import annotations

from aegismem.retrieval.bm25 import BM25Index, tokenize
from aegismem.retrieval.embeddings import HashingEmbedder, cosine
from aegismem.retrieval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter, Strategy, choose_strategy
from aegismem.retrieval.vector import VectorIndex

CORPUS = {
    "m_pg": "Primary database migrated to PostgreSQL 16 during the change window",
    "m_mysql": "Legacy checkout service used MySQL 8.0 before migration",
    "m_redis": "Cache layer runs Redis 7 with 2GB maxmemory",
    "m_host": "The metrics API is reachable at metrics.prod.internal:9090",
    "m_sop": "Runbook: on connection pool exhaustion, restart the service and scale replicas",
}


def _router(reranker: OverlapReranker | None = None) -> JITRouter:
    r = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), reranker, top_k=3)
    for doc_id, text in CORPUS.items():
        r.index(doc_id, text)
    return r


# -- metrics -----------------------------------------------------------------


def test_metric_functions() -> None:
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert recall_at_k(ranked, rel, 4) == 1.0
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert reciprocal_rank(ranked, rel) == 0.5
    assert 0.0 < ndcg_at_k(ranked, rel, 4) <= 1.0
    assert ndcg_at_k(["b", "d"], rel, 2) == 1.0


# -- bm25 --------------------------------------------------------------------


def test_bm25_ranks_exact_terms_first() -> None:
    idx = BM25Index()
    for doc_id, text in CORPUS.items():
        idx.add(doc_id, text)
    ranked = [d for d, _ in idx.search("postgresql change window")]
    assert ranked[0] == "m_pg"


def test_bm25_remove_updates_index() -> None:
    idx = BM25Index()
    idx.add("d1", "redis cache")
    idx.add("d2", "redis queue")
    idx.remove("d1")
    assert len(idx) == 1
    assert all(d == "d2" for d, _ in idx.search("redis"))


# -- embeddings --------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=128)
    v1 = emb.embed("postgres migration")
    v2 = emb.embed("postgres migration")
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9
    assert cosine(v1, emb.embed("database migration to postgres")) > cosine(
        v1, emb.embed("redis cache memory")
    )


# -- adaptive strategy -------------------------------------------------------


def test_strategy_selection() -> None:
    assert choose_strategy("metrics.prod.internal:9090") is Strategy.BM25_HEAVY
    assert choose_strategy("DB_PASSWORD=") is Strategy.BM25_HEAVY
    assert (
        choose_strategy("why did the checkout service start erroring after lunch")
        is Strategy.VECTOR_HEAVY
    )
    assert choose_strategy("postgres") is Strategy.FULL_HYBRID


# -- router behaviour --------------------------------------------------------


def test_router_retrieves_relevant_and_logs_strategy() -> None:
    router = _router()
    res = router.retrieve("what database did we migrate to")
    assert not res.empty
    assert res.results[0].memory_id == "m_pg"
    assert res.strategy in set(Strategy)
    assert res.latency_ms >= 0.0
    # quality beside latency:
    ranked = [r.memory_id for r in res.results]
    assert recall_at_k(ranked, {"m_pg"}, 3) == 1.0


def test_router_returns_empty_when_nothing_clears_threshold() -> None:
    router = _router()
    res = router.retrieve("kubernetes helm chart quantum blockchain")
    assert res.empty
    assert res.results == []


def test_router_respects_top_k() -> None:
    router = _router()
    res = router.retrieve("service migration database cache runbook restart")
    assert len(res.results) <= 3


def test_reranker_reorders_candidates() -> None:
    router = _router(OverlapReranker())
    res = router.retrieve("connection pool exhaustion restart", documents=CORPUS)
    assert not res.empty
    assert res.results[0].memory_id == "m_sop"
    assert res.results[0].rerank is not None


def test_mrr_across_queries() -> None:
    router = _router()
    queries = {
        "postgresql migration change window": "m_pg",
        "redis cache maxmemory": "m_redis",
        "metrics api host": "m_host",
    }
    rankings = []
    relevants = []
    for q, gold in queries.items():
        res = router.retrieve(q, documents=CORPUS)
        rankings.append([r.memory_id for r in res.results])
        relevants.append({gold})
    assert mean_reciprocal_rank(rankings, relevants) >= 0.8


def test_tokenize_basic() -> None:
    assert tokenize("Postgres-16, MySQL!") == ["postgres", "16", "mysql"]
