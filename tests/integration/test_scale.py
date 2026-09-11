"""Production Phase P5 gate — scale harness runs and quality holds as it grows.

Runs the scale harness at a small, CI-fast size and asserts retrieval quality
stays high while latency is actually measured (never asserted). The 100k-1M runs
against pgvector are invoked manually; this locks in the regression.
"""

from __future__ import annotations

from benchmarks.scale.run import measure


def test_scale_quality_holds_and_latency_measured() -> None:
    small = measure(500, n_queries=25)
    assert small.scale == 500
    assert small.recall_at_k >= 0.9  # distinctive facts are found even in a filler corpus
    assert small.mrr >= 0.9
    assert small.p95_ms >= small.p50_ms >= 0.0  # latency is measured, not asserted


def test_quality_does_not_collapse_with_scale() -> None:
    b = measure(2000, n_queries=25)
    # More filler must not tank recall of the labelled targets.
    assert b.recall_at_k >= 0.9
    assert b.mrr >= 0.85
