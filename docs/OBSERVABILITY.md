# OBSERVABILITY & REPLAY (Phase 7)

Observability here means **reproducibility, not just logging**. Every run records
enough to reconstruct *why* it behaved as it did — and to re-execute it
deterministically — via `src/aegismem/observability/`.

## Source of truth: the local trace store

An **append-only JSONL log** (`JSONLTraceStore`) is authoritative. One line per
record, discriminated by `kind`:

| Record | Captures |
|---|---|
| `RunRecord` | `request_id → run_id → trace_id`, session, input, status, usage (tokens, tool calls), total latency, outcome, error |
| `SpanRecord` | one per pipeline stage: name, order, duration, status, and the stage's **decision attributes** (retrieval strategy, retrieved IDs + scores, guardrail verdict, selected tools, budget usage, provider) |
| `CallRecord` | one per non-deterministic call (LLM / tool): request key + recorded response + latency, so the call can be served back on replay |

Writes are **fire-and-forget** (`failure_policy: observability: continue`): a
failed or broken sink increments a drop counter and the run proceeds. Tracing can
never take down a run — proven by `test_tracing_is_fail_open`.

## Langfuse is a derived exporter, not the truth

`LangfuseExporter` mirrors records into a Langfuse span-waterfall for the demo. It
is import-guarded (Langfuse is in the `observability` dependency group) and a
silent no-op when the package or config is absent. `FanoutSink` emits to the
local store and the exporter together, isolating a misbehaving exporter.

## Deterministic replay

```bash
uv run aegismem replay <run_id>              # human-readable pipeline
uv run aegismem replay <run_id> --json       # machine-readable
```

`Replayer.reconstruct` rebuilds the ordered timeline (guardrails → retrieval → …
→ execution → output) with each stage's attributes. `Replayer.cache(run_id)`
builds a `ReplayCache` of the recorded LLM/tool responses; attaching it via
`Tracer.run(request, cache=...)` makes the same calls return their **recorded**
outputs instead of re-executing — so non-deterministic agent behavior becomes
debuggable. Proven by `test_recorded_call_replays_without_reexecuting`.

## Wiring

```python
store = JSONLTraceStore("traces/trace.jsonl")
tracer = Tracer(FanoutSink([store, LangfuseExporter()]))
with tracer.run(request) as run:
    with run.span("retrieval") as sp:
        sp.set(strategy="hybrid", retrieved=[r.memory_id for r in res.results])
    answer = run.record_llm("plan", {"prompt": p}, produce=lambda: llm(p))
    run.set_output(answer, usage=Usage(input_tokens=..., output_tokens=...))
```

The trace directory is gitignored; at the documented single-node scale a flat
JSONL scan powers replay, with the same interface fronting a SQLite/pgvector store
on the production swap path.
