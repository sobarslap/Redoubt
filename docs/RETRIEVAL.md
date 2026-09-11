# Retrieval

The JIT router is hybrid **BM25 + vector**, with cross-encoder rerank, an
adaptive strategy, a relevance threshold, and top-K — measured, not asserted.

## Design

- **Hand-rolled BM25** (`retrieval/bm25.py`) — Okapi with the standard `k1`/`b`
  knobs; owned line-by-line rather than pulled from a library.
- **Embeddings** (`retrieval/embeddings.py`) — an `Embedder` seam. Production is
  `sentence-transformers bge-small-en-v1.5`; the zero-dependency
  `HashingEmbedder` is the default so the stack builds and benchmarks on any
  interpreter (notably Python 3.14, where ML wheels may lag).
- **Reranker** (`retrieval/reranker.py`) — a `Reranker` seam. Production is
  `bge-reranker-base`; `OverlapReranker` is the dependency-free default. When no
  reranker is supplied the router falls back to hybrid-score ordering (the
  documented failure policy).
- **Adaptive strategy** (`retrieval/router.py`) — the query shape selects
  weights and the choice is logged per run:
  - exact identifier / hostname / key → **BM25-heavy** (0.8 / 0.2)
  - conceptual question → **vector-heavy** (0.2 / 0.8)
  - ambiguous / high-value → **full hybrid** (0.5 / 0.5) + rerank

## The EMPTY MEMORY guarantee

If nothing clears the relevance threshold, the router returns **empty** rather
than inject a vaguely-related fact. Fusion mixes an unbounded BM25 score (min-max
scaled for *ordering only*) with an absolute cosine signal, so an **eligibility
gate** on absolute signals — a real lexical hit, or cosine ≥ `min_vector_sim` —
runs before scaling. This stops min-max from promoting near-zero similarity to a
"match". Empty is safer than wrong.

## Measured (reproducible)

`uv run python -m benchmarks.retrieval.run` — 8-memory labelled corpus, 7 queries,
`HashingEmbedder` + `OverlapReranker`:

| Metric | Value |
|---|---|
| Recall@3 | 1.000 |
| Precision@3 | 0.333¹ |
| NDCG@3 | 1.000 |
| MRR | 1.000 |
| empty-result rate | 0.000 |
| latency P50 / P95 | ~0.36 / ~0.41 ms |

¹ One gold memory per query against top-3, so Precision@3 caps at 1/3 by
construction — a property of the label set, not a ranking defect.

Numbers rise when the real `bge` embedder/reranker replace the hashed defaults;
the harness and gate stay identical. Scale-out figures (10 → 1M memories) land in
Phase 9 against `docs/PERFORMANCE.md`.
