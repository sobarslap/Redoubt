# EVALUATION & REGRESSION (Phase 8)

A **categorized golden dataset** run through deterministic, offline evaluators
over the *real* subsystems. No network, no API keys, no live LLM — reproducible at
~zero cost, and fast enough to gate every CI run. Regressions block the pipeline.

```bash
uv run python -m evals.run     # the CI eval gate (exit 1 on any breach)
```

## Dataset

`evals/golden_dataset/*.jsonl` — one JSONL file per category, one compact
scenario per line (`{id, category, description, params, expect}`). **107
scenarios** across all nine categories:

| Category | Drives | What it checks |
|---|---|---|
| `memory` | conflict/corruption | injected untrusted writes never land ACTIVE |
| `retrieval` | Recall / hit-rate | hybrid router finds the gold memory; empty when nothing clears the threshold |
| `compaction` | critical-state preservation | verified compaction keeps `critical_variables`, refuses lossy commits |
| `tools` | authorization | allow/deny/permission/operation gates decide, not the model |
| `security` | attack-success-rate | named injection payloads blocked; benign chatter not flagged |
| `long_context` | cost / preservation | extreme conversations compact within budget, critical state survives |
| `failure_recovery` | policy adherence | fail-open observability, empty-memory, fail-closed classifier, bounded tool timeout |
| `cost` | tokens/run | progressive tool-schema injection ≪ catalog; compaction reduction |
| `latency` | P50/P95 | retrieval P95 under budget |

## Harness & gates

`evals/harness.py` loads the dataset, dispatches each scenario to its category
evaluator (`evals/evaluators.py`), and aggregates a `Report`. The evaluators own
their subsystem fixtures so scenarios stay small; a crashing evaluator is a
failure, never a stack trace.

Gate (blocks CI):

- **per-category pass-rate == 1.0** — the golden scenarios encode expected
  behavior, so a regression flips one to fail.
- **cross-cutting invariants == 0** — `memory_corruption`,
  `unauthorized_tool_exec`, `attack_success`, aggregated across every scenario
  that touches them.

Current run: **107/107 passed**, all categories at gate, all invariants 0. The
same gate is asserted in `tests/e2e/test_golden_dataset.py` and run in CI after
the unit, integration, and adversarial suites.

## Relation to DeepEval

The harness is the deterministic, always-on gate. DeepEval (in the `evaluation`
dependency group) layers LLM-judged faithfulness/relevancy on top for the runs
that use a live provider; it is additive, never on the zero-cost critical path.
