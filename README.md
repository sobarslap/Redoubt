# AegisMem

> A framework-free (no LangChain / LlamaIndex), production-grade **agent runtime**
> in pure Python. Memory, retrieval, context, tools, security, observability, and
> evaluation are each independent, benchmarked subsystems with a deterministic
> execution contract.

**Status:** Stage 0 — project foundation. The runtime is scaffolded; subsystems
land phase by phase per [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md).

## Why

Production agents break on four failure modes: hallucination, latency,
prompt-injection/poisoning, and runaway token cost. AegisMem addresses each with
mechanisms that are *measurable, deterministic where possible, failure-aware,
secure, testable, and reproducible* — proven by a benchmark suite, not asserted.

## Quickstart (dev)

```bash
uv sync --group dev            # core runtime + dev tooling
uv run aegismem version
uv run aegismem config         # resolved settings + loaded policy
uv run pytest                  # Stage 0 gate: smoke tests green
uv run ruff check .
uv run mypy
```

Heavier subsystems install as dependency groups when their phase begins, e.g.
`uv sync --group retrieval --group llm`.

## Layout

```
src/aegismem/{api,memory,retrieval,context,mcp,guardrails,execution,observability,config}
demo/devops_sre/        # scripted DevOps/SRE incident (the story that sells it)
tests/{unit,integration,e2e,adversarial}
benchmarks/{retrieval,latency,tokens,compaction,security}
evals/golden_dataset/
```

## Docs

Architecture, security trust model, memory model, retrieval guarantees, MCP tool
discovery, evaluation, performance envelope, and threat model live in [`docs/`](docs/).
