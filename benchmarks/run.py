"""Benchmark orchestrator — prove every claim as a measured baseline→AegisMem delta.

    python -m benchmarks.run --scales 10,1000,100000,1000000 [--charts]

Runs the token/context, tool-schema, compaction, latency, and security suites,
prints a baseline-vs-AegisMem table, and writes ``benchmarks/results/summary.csv``.
With ``--charts`` (needs the ``bench`` dependency group) it also renders PNGs. No
LLM, no network — the whole thing is reproducible at ~zero cost, and the CV's
``X% / Y%`` numbers come straight from here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmarks.common import parse_scales
from benchmarks.compaction.run import compaction_point
from benchmarks.latency.run import measure as measure_latency
from benchmarks.tokens.run import schema_comparison, token_comparison

_RESULTS = Path(__file__).parent / "results"
_DEFAULT_SCALES = [10, 1000, 100_000, 1_000_000]


def _security_asr() -> tuple[float, float]:
    from tests.adversarial.corpus import ATTACKS, BENIGN

    from aegismem.guardrails.injection import PromptInjectionScanner

    scanner = PromptInjectionScanner()
    missed = sum(1 for a in ATTACKS if not scanner.scan(a.payload).blocked)
    fp = sum(1 for t in BENIGN if scanner.scan(t).blocked)
    return missed / len(ATTACKS), fp / len(BENIGN)


def run(scales: list[int], *, charts: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for scale in scales:
        c = token_comparison(scale)
        rows.append(
            {
                "suite": "context_tokens",
                "scale": scale,
                "baseline": c.baseline_tokens,
                "aegismem": c.aegismem_tokens,
                "reduction": f"{c.reduction:.6f}",
            }
        )

    for n in [10, 50, 200]:
        s = schema_comparison(n)
        rows.append(
            {
                "suite": "tool_schema_chars",
                "scale": n,
                "baseline": s.baseline_chars,
                "aegismem": s.aegismem_chars,
                "reduction": f"{s.reduction:.6f}",
            }
        )

    for turns in [20, 50, 100, 200]:
        p = compaction_point(turns)
        rows.append(
            {
                "suite": "compaction_tokens",
                "scale": turns,
                "baseline": p.tokens_before,
                "aegismem": p.tokens_after,
                "reduction": f"{p.reduction:.6f}",
                "note": f"critical_preserved={p.critical_preserved}",
            }
        )

    lat = measure_latency()
    rows.append(
        {
            "suite": "retrieval_latency_ms",
            "scale": lat.queries,
            "baseline": "",
            "aegismem": lat.p95_ms,
            "reduction": "",
            "note": f"p50={lat.p50_ms}",
        }
    )

    asr, fpr = _security_asr()
    rows.append(
        {
            "suite": "attack_success_rate",
            "scale": "",
            "baseline": "",
            "aegismem": f"{asr:.6f}",
            "reduction": "",
            "note": f"false_positive_rate={fpr:.6f}",
        }
    )

    _write_csv(rows)
    _print_table(scales, asr, fpr, lat.p50_ms, lat.p95_ms)
    if charts:
        _render_charts(scales)
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    _RESULTS.mkdir(exist_ok=True)
    fields = ["suite", "scale", "baseline", "aegismem", "reduction", "note"]
    out = _RESULTS / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nwrote {out}")


def _print_table(scales: list[int], asr: float, fpr: float, p50: float, p95: float) -> None:
    print("context tokens - naive dump-all vs AegisMem top-K:")
    print(f"{'scale':>12} {'baseline':>16} {'aegismem':>10} {'reduction':>10}")
    for scale in scales:
        c = token_comparison(scale)
        print(f"{c.scale:>12} {c.baseline_tokens:>16} {c.aegismem_tokens:>10} {c.reduction:>9.2%}")
    print(f"\nretrieval latency  P50 {p50:.3f} ms  P95 {p95:.3f} ms")
    print(f"attack-success-rate {asr:.3f} (target 0)   false-positive-rate {fpr:.3f}")


def _render_charts(scales: list[int]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("charts skipped: matplotlib not installed (uv sync --group bench)")
        return
    xs = scales
    baseline = [token_comparison(s).baseline_tokens for s in xs]
    aegis = [token_comparison(s).aegismem_tokens for s in xs]
    fig, ax = plt.subplots()
    ax.plot(xs, baseline, marker="o", label="naive (dump-all)")
    ax.plot(xs, aegis, marker="o", label="AegisMem (top-K)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("memories in store")
    ax.set_ylabel("context tokens")
    ax.set_title("Context tokens: naive baseline vs AegisMem")
    ax.legend()
    _RESULTS.mkdir(exist_ok=True)
    path = _RESULTS / "context_tokens.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmarks.run", description="AegisMem benchmarks")
    parser.add_argument(
        "--scales", default="10,1000,100000,1000000", help="comma-separated memory-store sizes"
    )
    parser.add_argument(
        "--charts", action="store_true", help="render PNG charts (needs bench group)"
    )
    args = parser.parse_args()
    run(parse_scales(args.scales) or _DEFAULT_SCALES, charts=args.charts)


if __name__ == "__main__":
    main()
