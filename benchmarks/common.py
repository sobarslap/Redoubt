"""Shared benchmark primitives — the naive baseline and the AegisMem model.

The claims Phase 9 proves are *deltas*: naive-agent vs AegisMem, measured, never
asserted. The naive baseline is the strawman every framework demo ships —
**dump all memory into context, inject all tool schemas, never compact**. AegisMem
retrieves top-K, injects only matched schemas, and compacts at the watermark.

Token counts use the same :class:`HeuristicTokenCounter` the runtime uses, over
representative SRE-memory text. Scale-dependent metrics are computed from a
measured per-memory mean rather than materializing 1M objects — the mean is
measured on a real generated pool, so the extrapolation is honest and O(1) in the
scale. The methodology is stated in ``docs/PERFORMANCE.md``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from aegismem.context.tokens import HeuristicTokenCounter

_COUNTER = HeuristicTokenCounter()

# A representative pool of SRE memory lines; the per-memory token mean is measured
# from this pool and used to extrapolate to large scales.
_POOL: tuple[str, ...] = (
    "Primary database migrated to PostgreSQL 16 during the 2024-03 change window",
    "Legacy checkout service used MySQL 8.0 before the migration was completed",
    "Cache layer runs Redis 7 with 2GB maxmemory and an LRU eviction policy",
    "The metrics API is reachable at metrics.prod.internal:9090 over mTLS",
    "Runbook: on connection pool exhaustion restart the service and scale replicas",
    "PagerDuty alert fired for checkout p95 latency above the 800ms SLO at 03:14Z",
    "Deploy cadence is weekly on Thursdays via the blue-green pipeline in us-east-1",
    "The payments service is owned by the transactions platform team, oncall rotation weekly",
    "TLS certificates for the gateway renew automatically via cert-manager every 60 days",
    "The read replica lag threshold before an automated failover is 30 seconds",
)

MEAN_TOKENS_PER_MEMORY: float = statistics.mean(_COUNTER.count(m) for m in _POOL)

# Fixed AegisMem context components (tokens), independent of memory-store size.
SYSTEM_PROMPT_TOKENS = 220
RETRIEVAL_TOP_K = 5


@dataclass(frozen=True)
class TokenComparison:
    scale: int
    baseline_tokens: int  # naive: every memory dumped into context
    aegismem_tokens: int  # retrieved top-K + fixed overhead
    reduction: float  # 1 - aegismem/baseline


def token_comparison(scale: int, *, top_k: int = RETRIEVAL_TOP_K) -> TokenComparison:
    baseline = round(scale * MEAN_TOKENS_PER_MEMORY)
    # Retrieval can return at most `scale` memories, so a small store (scale < top_k,
    # e.g. --scales 1) caps the retrieved count and can't overcount aegismem tokens.
    aegismem = round(SYSTEM_PROMPT_TOKENS + min(scale, top_k) * MEAN_TOKENS_PER_MEMORY)
    reduction = 1.0 - (aegismem / baseline) if baseline else 0.0
    return TokenComparison(scale, baseline, aegismem, round(reduction, 6))


def p50_p95(latencies: list[float]) -> tuple[float, float]:
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    # Nearest-rank P95: ceil(0.95*n) as a 1-based rank, then 0-index. Flooring
    # (int(0.95*n)) underreports whenever n is not a multiple of 20 (e.g. n=21
    # would pick rank 19 instead of 20).
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
    return p50, p95


def parse_scales(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]
