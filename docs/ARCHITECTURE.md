# ARCHITECTURE

AegisMem is an **agent runtime with a deterministic execution contract**, not a
script. Every stage has typed input, typed output, a single typed error envelope,
timing, and trace metadata. Subsystems are independent modules composed by the
runtime (`src/aegismem/execution/agent.py`); each is separately testable and
benchmarked.

## The execution contract

```
AgentRequest{request_id, session_id, input, mode, idempotency_key, budget_overrides?}
  → Validation        Pydantic schema + size limits
  → Guardrails        trust-label input, injection scan, PII scan
  → Budget allocation retrieval + context + tool budgets fixed for the run
  → Retrieval         JIT hybrid BM25+vector+rerank, relevance threshold, top-K
  → Execution         provider-agnostic LLM reasoning over labeled, fenced context
  → Output validation grounding/citation, secret + PII leakage filter
  → AgentResponse{run_id, output, citations[], usage, timings, trace_id, status, error?}
```

**Correlation** threads the whole pipeline: `request_id` (client) → `run_id`
(engine) → `trace_id` (observability), on every span and log line. **Errors** are
one envelope `{code, category, message, retryable, stage}` — never a raw stack.

## Subsystems

| Module | Responsibility | Key invariant |
|---|---|---|
| `memory` | 5-tier records, lifecycle state machine, deterministic conflict resolution, provenance graph | memory corruption = 0 |
| `retrieval` | adaptive BM25+vector+rerank router with a relevance threshold | empty memory over vague memory |
| `context` | budget allocator, token watchdog, verified compaction | critical state provably preserved |
| `mcp` | progressive tool discovery + the guarded Execute_Tool gateway | discovery ≠ authorization; unauthorized exec = 0 |
| `guardrails` | trust boundary (fenced untrusted data), injection scanner, PII, sanitization | attack-success-rate → 0, fail-closed |
| `execution` | agent loop + provider-agnostic `LLMClient` (Gemini/Claude/Ollama/Mock) with retry+fallback | provider-swappable, keyless demo path |
| `observability` | append-only trace store + deterministic replay; optional Langfuse | tracing fail-open; a run with fully persisted traces replays from `run_id` |
| `config` | budgets, policies, model routing as YAML | thresholds are configuration, not literals |

## Storage — the source of truth

```
SQLite (WAL, authoritative memory)  Vector index (derived)      BM25 index (derived)
├── memory rows + metadata          └── embedding → memory_id   └── postings → memory_id
├── version history / lineage
└── status / provenance graph

JSONL trace store (authoritative observability, separate from the memory DB)
└── run / trace / replay records   (fail-open: a dropped write is not replayable)
```

The vector and lexical indexes are **rebuildable accelerators**; the memory SQLite
owns the truth for memory, and the append-only JSONL trace store owns run/trace
records. They are distinct stores — traces are not written into the memory DB. `MemoryStore`, `LLMClient`, and `TraceSink` are `Protocol`s — the
documented swap points to a distributed backend (pgvector / Qdrant), a real LLM
provider, or a hosted trace sink, with no change to the runtime.

## Failure posture

Per-component policy (`config/policies.yaml → failure_policy`): observability
**continues** (fire-and-forget), retrieval **degrades** to no-memory mode, the
security classifier **fails closed**, LLM requests **retry then fall back** across
providers, compaction **preserves previous** context on verification failure. Each
policy has a test that injects the failure and asserts the behavior.

## Design decisions

- **Framework-free** (no LangChain/LlamaIndex) — line-by-line control is the point.
- **LLM-assists, never LLM-authoritative** — conflict resolution, tool
  authorization, and compaction commits are decided by deterministic policy
  layers; the LLM only proposes.
- **Deterministic and offline by default** — every guarantee is provable at ~zero
  cost; real providers and heavy models are opt-in dependency groups.
