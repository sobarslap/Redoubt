"""Eval harness — load the golden dataset, run every scenario through its
evaluator, and aggregate into a gate-able :class:`Report`.

Deterministic and offline: no network, no API keys. CI runs ``python -m
evals.run``; a per-category pass-rate below its gate, or any nonzero
cross-cutting invariant, fails the build.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from evals.evaluators import EVALUATORS
from evals.schema import Category, CategoryReport, EvalResult, Report, Scenario

_DATASET_DIR = Path(__file__).parent / "golden_dataset"

# Regression gate: the golden scenarios encode expected behavior, so every one
# must pass. A regression flips one to fail and blocks CI.
_GATES: dict[Category, float] = {c: 1.0 for c in Category}


def load_dataset(root: Path | None = None) -> list[Scenario]:
    root = root or _DATASET_DIR
    scenarios: list[Scenario] = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                scenarios.append(Scenario.model_validate_json(line))
    return scenarios


def run(scenarios: list[Scenario]) -> Report:
    results: list[EvalResult] = []
    for s in scenarios:
        evaluator = EVALUATORS[s.category]
        try:
            results.append(evaluator(s))
        except Exception as exc:  # a crashing evaluator is a failure, not a stack trace
            results.append(
                EvalResult(
                    scenario_id=s.id,
                    category=s.category,
                    passed=False,
                    score=0.0,
                    detail=f"evaluator raised: {type(exc).__name__}: {exc}",
                )
            )
    return aggregate(results)


def aggregate(results: list[EvalResult]) -> Report:
    by_cat: dict[Category, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    cats: list[CategoryReport] = []
    for cat in Category:
        rs = by_cat.get(cat, [])
        if not rs:
            continue
        cats.append(
            CategoryReport(
                category=cat,
                total=len(rs),
                passed=sum(1 for r in rs if r.passed),
                gate=_GATES[cat],
            )
        )

    return Report(
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        categories=cats,
        memory_corruption=sum(r.memory_corruption for r in results),
        unauthorized_tool_exec=sum(r.unauthorized_tool_exec for r in results),
        attack_success=sum(r.attack_success for r in results),
        failures=[
            f"{r.category.value}/{r.scenario_id}: {r.detail}" for r in results if not r.passed
        ],
    )


def run_dataset(root: Path | None = None) -> Report:
    return run(load_dataset(root))
