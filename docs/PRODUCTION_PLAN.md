# AegisMem — Production Hardening Plan

> The [build plan](BUILD_PLAN.md) is complete: all 10 phases landed, the runtime
> is proven end to end at ~zero cost. This plan is the **separate, optional track**
> that takes AegisMem from *"impressive, reproducible prototype"* to *"a live
> service under real load with real data and real (cost-incurring) LLM calls."*
>
> Same discipline as the build plan: **vertical, phase-gated slices**; never start
> a phase before the prior gate passes; every claim measured, not asserted. The
> effort figures are *human full-time-equivalent* estimates so you can budget —
> delivered here as phase-gated sessions, additively, with **no rewrites** (the
> `MemoryStore` / `LLMClient` / `TraceSink` protocols are the designed swap points).

## Scope decision — pick your finish line first

The total is a **4–8 week** range because two of these are genuinely optional. Decide up front:

- **Lean service (~4 wks):** P1–P4 + P7. A solid single-tenant deployed service on a real backend and a real LLM.
- **Full production (~8 wks):** all phases, including multi-tenant scale testing (P5), live security audit (P6), and the full observability/ops story (P8).

---

## Phase P1 — Real LLM providers, wired and integration-tested
**~3–5 days.** The `LLMClient` adapters (Gemini/Claude/Ollama) exist but have never
touched a live API. Wire real calls behind the existing seam.

- Real `generate_content` / `messages.create` paths, streaming, real token accounting from provider usage.
- Model routing from `config/models.yaml` roles (workhorse/reasoning/judge/offline) resolved by a factory.
- Cost controls: per-run token ceiling, spend cap, request timeouts, real retry/backoff tuned to provider 429s.
- Function/tool-calling adapter so the LLM can actually drive `Execute_Tool` (today the demo scripts it).
- **Gate:** the DevOps demo passes end to end with `LLM_PROVIDER=gemini`, `=claude`, and `=ollama`; a live integration test (network-gated, keyed) is green; token/cost accounting matches provider dashboards within tolerance.

## Phase P2 — Distributed memory backend behind `MemoryStore`
**~5–8 days.** Swap SQLite + in-proc indexes for a networked store without changing the runtime.

- `PgVectorMemoryStore` (Postgres + pgvector) — or Qdrant — implementing the exact `MemoryStore` protocol.
- Schema migrations (Alembic), connection pooling, the provenance/lineage tables at scale, WAL→transactional semantics.
- Real embedding + reranker models from the `retrieval` group (`bge-small`, `bge-reranker`) replacing the hashing/overlap dev stand-ins; batch embedding pipeline.
- Index build/rebuild jobs (the derived accelerators) with a documented rebuild path.
- **Gate:** the full unit/integration/adversarial/eval suites pass against the new backend unchanged; a parity test asserts identical retrieval results (within tolerance) between SQLite and pgvector on the golden corpus.

## Phase P3 — The service: FastAPI async runtime
**~4–6 days.** Turn the library + CLI into a networked service honoring the runtime contract.

- FastAPI app: `POST /runs` (sync + async modes), `GET /runs/{run_id}`, `/memory/{id}/provenance`, `/healthz`, `/readyz`.
- Async lifecycle, per-stage deadlines, cooperative cancellation at stage boundaries, `idempotency_key` dedupe.
- Structured request/response using the existing `AgentRequest`/`AgentResponse`; the typed error envelope on every path.
- Backpressure and concurrency limits; graceful shutdown flushing traces.
- **Gate:** load-safe under concurrent requests (no shared-state races); contract tests for sync/async/idempotency/cancellation; OpenAPI schema published.

## Phase P4 — AuthN/Z, secrets, tenancy
**~4–6 days.** Nobody unauthenticated touches the service; secrets never live in code.

- API-key or OIDC auth on the service; per-caller rate limiting and quotas.
- Secrets via env/secret-manager (never in `models.yaml`); provider keys injected at runtime.
- Session/tenant isolation in memory (a tenant can never retrieve another's memories) — enforced in `MemoryStore` queries, tested adversarially.
- Audit log of privileged actions (tool executions, memory deletes, policy changes).
- **Gate:** an authz test suite proves cross-tenant isolation and that unauthenticated/over-quota requests are refused with typed errors; a secrets scan (gitleaks) is clean in CI.

## Phase P5 — Scale & performance validation (optional, "full" track)
**~4–6 days.** Prove the benchmarked deltas hold on the real backend at real scale.

- Seed 100k → 1M real memories into pgvector; measure retrieval Recall/MRR + P50/P95 at each scale (not the analytic extrapolation — the real thing).
- Load test the service (Locust/k6): throughput, tail latency, saturation point, the documented single-node → sharded boundary.
- Cost-per-run at scale with the live LLM; tune budgets.
- **Gate:** `docs/PERFORMANCE.md` gains a *measured-at-scale* section replacing the extrapolation; P95 and cost stay within stated budgets; the regression benchmark runs in CI against a scaled fixture.

## Phase P6 — Live security audit & red-team (optional, "full" track)
**~3–5 days.** The Phase-6 red-team scored the deterministic scanner; now score the *live LLM path*.

- Run the AI-Infra-Guard attack suite against the real model end to end (not just the classifier): direct/indirect injection, ActorAttack/PAIR/GOAT multi-turn, memory poisoning, exfiltration.
- Measure real attack-success-rate through the full boundary + live model; fix gaps; add the optional HF PI classifier (`deberta-v3-prompt-injection`) as defense-in-depth.
- Dependency/CVE scanning, SBOM, threat-model review against the deployed topology.
- **Gate:** live attack-success-rate at/near 0 with a measured false-positive rate; `docs/SECURITY.md` gains a live-results section; no high-severity CVEs.

## Phase P7 — Observability & ops
**~3–5 days.** From local JSONL traces to a running observability stack.

- Self-hosted Langfuse (Docker) as the `TraceSink` fanout target; retention + PII-safe trace redaction.
- Metrics (Prometheus) + dashboards + alerting on the eval/security/latency/cost gates (a risen attack-success-rate pages someone).
- Deterministic replay wired to the hosted store; `aegismem replay` works against production runs.
- **Gate:** a production run appears as a Langfuse waterfall and replays by `run_id`; alerts fire on injected gate breaches in staging.

## Phase P8 — Deploy, CI/CD, release
**~3–5 days.** One command ships it; regressions block the ship.

- Containerize (multi-stage Docker), IaC for the service + Postgres/Qdrant + Langfuse, staging + prod environments.
- CI/CD extends the existing pipeline: format → lint → type → unit → integration → security → adversarial → **eval** → **perf** → build image → deploy staging → smoke → promote. A dropped eval/security/latency/cost gate blocks deploy.
- Versioned releases, migration runbook, rollback procedure, on-call runbook (dogfooding the DevOps/SRE demo's own domain).
- **Gate:** a green pipeline deploys to staging automatically; a deliberately-regressed eval score blocks promotion; rollback tested.

---

## Effort summary

| Track | Phases | Human FTE estimate |
|---|---|---|
| Lean deployed service | P1, P2, P3, P4, P7 | **~4 weeks** |
| Full production | P1–P8 | **~8 weeks** |

Each phase lands as its own gated session with tests, a doc update, and a
benchmark/scan where relevant — the same cadence as the build plan. Nothing here
reopens the runtime core; it all plugs into the interfaces already in place.
