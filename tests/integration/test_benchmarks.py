"""Phase 9 gate — the benchmark suite runs and the claimed deltas hold.

Not a perf-number regression (those live in benchmarks/results/), but a proof
that every headline claim is real and reproducible: AegisMem's context tokens are
near-constant while the naive baseline grows with the store, tool-schema overhead
shrinks, compaction reduces tokens while keeping critical state, and
attack-success-rate is zero.
"""

from __future__ import annotations

from benchmarks.compaction.run import compaction_point
from benchmarks.run import _security_asr, run
from benchmarks.tokens.run import schema_comparison, token_comparison


def test_context_tokens_reduction_grows_with_scale() -> None:
    small = token_comparison(1_000)
    large = token_comparison(1_000_000)
    # AegisMem's context is near-constant; the baseline scales linearly.
    assert large.aegismem_tokens == small.aegismem_tokens
    assert large.baseline_tokens > 100 * small.baseline_tokens
    assert large.reduction > 0.99


def test_tool_schema_overhead_shrinks() -> None:
    s = schema_comparison(200, top_k=3)
    assert s.aegismem_chars < s.baseline_chars
    assert s.reduction > 0.8  # top-3 of 200 schemas is a small fraction


def test_compaction_reduces_tokens_and_keeps_critical_state() -> None:
    p = compaction_point(100)
    assert p.committed
    assert p.critical_preserved
    assert p.reduction > 0.5


def test_attack_success_rate_is_zero() -> None:
    asr, fpr = _security_asr()
    assert asr == 0.0
    assert fpr <= 0.1


def test_orchestrator_runs_and_writes_summary(tmp_path, monkeypatch) -> None:
    import benchmarks.run as br

    monkeypatch.setattr(br, "_RESULTS", tmp_path)
    rows = run([10, 1000], charts=False)
    assert (tmp_path / "summary.csv").exists()
    suites = {r["suite"] for r in rows}
    assert {
        "context_tokens",
        "tool_schema_chars",
        "compaction_tokens",
        "retrieval_latency_ms",
        "attack_success_rate",
    } <= suites
