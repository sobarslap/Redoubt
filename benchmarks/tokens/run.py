"""Token / context-size benchmark — the headline "cut context tokens by X%" claim.

Compares the naive baseline (dump every memory into context) against AegisMem
(retrieve top-K + fixed overhead) at each memory scale, and the tool-schema
overhead (all schemas vs progressive top-K injection). Emits rows a caller can
aggregate; run standalone to print the table.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegismem.mcp.registry import ToolPermission, ToolRegistry, ToolSpec
from benchmarks.common import token_comparison


@dataclass(frozen=True)
class SchemaComparison:
    n_tools: int
    baseline_chars: int  # all schemas injected
    aegismem_chars: int  # top-K matched schemas injected
    reduction: float


def _registry(n_tools: int) -> ToolRegistry:
    reg = ToolRegistry()
    verbs = ["query", "read", "restart", "scale", "drain", "snapshot", "rotate", "flush"]
    nouns = ["metrics", "logs", "service", "replica", "cache", "certs", "queue", "index"]
    for i in range(n_tools):
        v, n = verbs[i % len(verbs)], nouns[(i // len(verbs)) % len(nouns)]
        reg.register(
            ToolSpec(
                name=f"{v}_{n}_{i}",
                description=f"{v} the {n} for a service over a window, with filters and options",
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "target service name"},
                        "window_min": {"type": "integer", "description": "lookback window"},
                        "filter": {"type": "string", "description": "optional filter expression"},
                    },
                    "required": ["service"],
                },
                handler=lambda a: "ok",
                permission=ToolPermission.ALLOW,
            )
        )
    return reg


def schema_comparison(n_tools: int, *, top_k: int = 3) -> SchemaComparison:
    reg = _registry(n_tools)
    baseline = reg.catalog_schema_chars()
    injected = reg.inject_schemas("metrics for a service", top_k=top_k)
    aegismem = sum(len(str(s)) for s in injected)
    reduction = 1.0 - (aegismem / baseline) if baseline else 0.0
    return SchemaComparison(n_tools, baseline, aegismem, round(reduction, 6))


def main(scales: list[int] | None = None) -> None:
    scales = scales or [10, 1000, 100_000, 1_000_000]
    print("context tokens - naive (dump-all) vs AegisMem (top-K):")
    print(f"{'scale':>10} {'baseline':>14} {'aegismem':>10} {'reduction':>10}")
    for scale in scales:
        c = token_comparison(scale)
        print(f"{c.scale:>10} {c.baseline_tokens:>14} {c.aegismem_tokens:>10} {c.reduction:>9.2%}")

    print("\ntool-schema overhead - all-schemas vs progressive injection:")
    print(f"{'tools':>10} {'baseline':>14} {'aegismem':>10} {'reduction':>10}")
    for n in [10, 50, 200]:
        s = schema_comparison(n)
        print(f"{n:>10} {s.baseline_chars:>14} {s.aegismem_chars:>10} {s.reduction:>9.2%}")


if __name__ == "__main__":
    main()
