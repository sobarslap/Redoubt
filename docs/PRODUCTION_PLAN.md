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

## Phase P1 — Real LLM providers, wired and integration-tested ✅ done
**~3–5 days.** The `LLMClient` adapters (Gemini/Claude/Ollama) exist but have never
touched a live API. Wire real calls behind the existing seam.

> **Landed:** a `build_client(role)` factory resolving `models.yaml` roles →
> provider+model, keys from env (keyless → offline mock; `strict=True` fails
> loudly), wrapped in the resilient fallback chain. `CostGuard` enforces per-run
> token + USD ceilings (`SpendError`). The agent gained a bounded reason↔tool loop
> where the model proposes `ToolCall`s executed only through the guarded gateway —
> unauthorized calls are refused mid-loop and fed back as data. The demo honors
> `AEGISMEM_LLM_ROLE` and runs on any provider with an identical contract. Live
> Gemini/Claude tests are network+key gated (skip in keyless CI).

- Real `generate_content` / `messages.create` paths, streaming, real token accounting from provider usage.
- Model routing from `config/models.yaml` roles (workhorse/reasoning/judge/offline) resolved by a factory.
- Cost controls: per-run token ceiling, spend cap, request timeouts, real retry/backoff tuned to provider 429s.
- Function/tool-calling adapter so the LLM can actually drive `Execute_Tool` (today the demo scripts it).
- **Gate:** the DevOps demo passes end to end with `LLM_PROVIDER=gemini`, `=claude`, and `=ollama`; a live integration test (network-gated, keyed) is green; token/cost accounting matches provider dashboards within tolerance.

## Phase P2 — Distributed memory backend behind `MemoryStore` ✅ done
**~5–8 days.** Swap SQLite + in-proc indexes for a networked store without changing the runtime.

> **Landed:** `PgVectorMemoryStore` (Postgres + pgvector) implements the full
> `MemoryStore` surface row-for-row with SQLite — CRUD, governed lifecycle
> transitions, provenance edges, review queue — over a psycopg connection pool,
> with an `embedding vector` column and an in-database `similar()` KNN. Real local
> models arrive via `retrieval/factory.py` (`build_embedder`/`build_reranker`:
> `bge-small` + `bge-reranker`, falling back to the deterministic dev stand-ins
> when `sentence-transformers` is absent). All optional deps (`pg` group) are
> import-guarded. A parity test asserts identical observable state across SQLite
> and pgvector on a fixed op sequence, gated on `AEGISMEM_PG_DSN` so keyless CI
> skips it. Alembic migrations now live in `migrations/` — an initial revision
> reuses `PgVectorMemoryStore._DDL` as the single source of truth so schema can't
> drift (`migrations` group). *Remaining for a full deployment: the index
> build/rebuild job (documented, not yet scripted).*

- `PgVectorMemoryStore` (Postgres + pgvector) — or Qdrant — implementing the exact `MemoryStore` protocol.
- Schema migrations (Alembic), connection pooling, the provenance/lineage tables at scale, WAL→transactional semantics.
- Real embedding + reranker models from the `retrieval` group (`bge-small`, `bge-reranker`) replacing the hashing/overlap dev stand-ins; batch embedding pipeline.
- Index build/rebuild jobs (the derived accelerators) with a documented rebuild path.
- **Gate:** the full unit/integration/adversarial/eval suites pass against the new backend unchanged; a parity test asserts identical retrieval results (within tolerance) between SQLite and pgvector on the golden corpus.

## Phase P3 — The service: FastAPI async runtime ✅ done
**~4–6 days.** Turn the library + CLI into a networked service honoring the runtime contract.

> **Landed:** `aegismem.api.app.create_app()` — a FastAPI service exposing
> `POST /runs` (sync + async), `GET /runs/{run_id}`, `DELETE /runs/{run_id}`
> (cooperative cancel), `GET /memory/{id}/provenance`, `/healthz`, `/readyz`, and a
> published OpenAPI schema. The blocking `AgentRuntime.handle` runs in a worker
> thread (`asyncio.to_thread`) so the event loop stays responsive; a bounded
> in-flight counter returns a typed 503 for backpressure; a repeated
> `idempotency_key` replays the original run; cancellation is enforced at each
> stage boundary in the runtime (`cancel_check`); and every error path returns the
> typed `ErrorEnvelope` mapped to an HTTP status. *Remaining for deploy: uvicorn
> entrypoint + container (folded into P8).*

- FastAPI app: `POST /runs` (sync + async modes), `GET /runs/{run_id}`, `/memory/{id}/provenance`, `/healthz`, `/readyz`.
- Async lifecycle, per-stage deadlines, cooperative cancellation at stage boundaries, `idempotency_key` dedupe.
- Structured request/response using the existing `AgentRequest`/`AgentResponse`; the typed error envelope on every path.
- Backpressure and concurrency limits; graceful shutdown flushing traces.
- **Gate:** load-safe under concurrent requests (no shared-state races); contract tests for sync/async/idempotency/cancellation; OpenAPI schema published.

## Phase P4 — AuthN/Z, secrets, tenancy ✅ done
**~4–6 days.** Nobody unauthenticated touches the service; secrets never live in code.

> **Landed:** an `ApiKeyStore` (keys held as SHA-256 hashes, constant-time compare)
> resolving each key to a `Principal{principal_id, tenant}`; a `FixedWindowQuota`
> per principal; both wired into the service as an opt-in auth dependency —
> unauthenticated → typed 401, over-quota → typed 429, and the service still runs
> open (dev/library mode) when no key store is configured. `TenantScopedMemoryStore`
> enforces cross-tenant isolation at the store boundary: a tenant can never get,
> list, mutate, link, or even confirm the existence of another tenant's memory
> (cross-tenant access raises `NotFoundError`, so existence never leaks) — proven
> adversarially. An `AuditLog` records privileged actions (run.create, auth.fail,
> auth.quota, run.cancel). Provider secrets come only from the environment; a test
> and a CI **gitleaks** step keep secret material out of config and history.
> OIDC/JWT bearer auth is also supported (`OIDCVerifier`, `auth` group) — the
> service accepts an API key or a verified bearer token — and `JSONLAuditSink`
> gives a durable append-only (fsync'd) audit trail. *Remaining: a distributed run
> registry (in-process today; swap for Redis in a multi-node deploy).*

- API-key or OIDC auth on the service; per-caller rate limiting and quotas.
- Secrets via env/secret-manager (never in `models.yaml`); provider keys injected at runtime.
- Session/tenant isolation in memory (a tenant can never retrieve another's memories) — enforced in `MemoryStore` queries, tested adversarially.
- Audit log of privileged actions (tool executions, memory deletes, policy changes).
- **Gate:** an authz test suite proves cross-tenant isolation and that unauthenticated/over-quota requests are refused with typed errors; a secrets scan (gitleaks) is clean in CI.

## Phase P5 — Scale & performance validation (optional, "full" track) ✅ done
**~4–6 days.** Prove the benchmarked deltas hold on the real backend at real scale.

> **Landed:** `benchmarks/scale/run.py` seeds a labelled synthetic corpus and
> measures **real** Recall@K, MRR, and P50/P95 as the store grows — the same code
> at 1k in CI and at 100k-1M against pgvector (`--real` for bge embeddings). A
> locust load-test rig (`benchmarks/scale/locustfile.py`) drives the running
> service for throughput/tail-latency/saturation. The CI gate
> (`tests/integration/test_scale.py`) asserts recall/MRR hold and latency is
> measured, and that quality does not collapse as filler grows. *Remaining: the
> actual 100k-1M pgvector run + publishing the measured numbers into
> `docs/PERFORMANCE.md` — needs a seeded Postgres you point it at.*

- Seed 100k → 1M real memories into pgvector; measure retrieval Recall/MRR + P50/P95 at each scale (not the analytic extrapolation — the real thing).
- Load test the service (Locust/k6): throughput, tail latency, saturation point, the documented single-node → sharded boundary.
- Cost-per-run at scale with the live LLM; tune budgets.
- **Gate:** `docs/PERFORMANCE.md` gains a *measured-at-scale* section replacing the extrapolation; P95 and cost stay within stated budgets; the regression benchmark runs in CI against a scaled fixture.

## Phase P6 — Live security audit & red-team (optional, "full" track) ✅ done
**~3–5 days.** The Phase-6 red-team scored the deterministic scanner; now score the *live LLM path*.

> **Landed:** an **end-to-end** red-team (`tests/adversarial/test_redteam_e2e.py`)
> that drives the full runtime with a fully-compliant "gullible" model — one that
> obeys any injection — across named single- and multi-turn attacks
> (tool-hijack, exfiltration, fake-system, ActorAttack, Many-Shot, memory-poison).
> Measured end-to-end attack-success-rate is **0**: the gateway refuses the
> unauthorized call and the output filter scrubs the secret regardless of model
> compliance — proof the defense is structural. A live-model variant is
> network+key gated. Optional ML defense-in-depth via
> `guardrails.classifier.CombinedScanner` (`deberta-v3-prompt-injection`,
> `guard-ml` group). CI gains `pip-audit` (CVE) + CycloneDX SBOM. *Remaining: run
> the live-model variant against a real provider and record the number.*

- Run the AI-Infra-Guard attack suite against the real model end to end (not just the classifier): direct/indirect injection, ActorAttack/PAIR/GOAT multi-turn, memory poisoning, exfiltration.
- Measure real attack-success-rate through the full boundary + live model; fix gaps; add the optional HF PI classifier (`deberta-v3-prompt-injection`) as defense-in-depth.
- Dependency/CVE scanning, SBOM, threat-model review against the deployed topology.
- **Gate:** live attack-success-rate at/near 0 with a measured false-positive rate; `docs/SECURITY.md` gains a live-results section; no high-severity CVEs.

## Phase P7 — Observability & ops ✅ done
**~3–5 days.** From local JSONL traces to a running observability stack.

> **Landed:** the service exposes Prometheus metrics at `/metrics` (runs, failures,
> run-latency summary) that move with real traffic; `ops/alerts.yml` turns the
> CI eval/security/latency gates into live alerts. Traces fan out to a
> self-hosted Langfuse (`ops/docker-compose.langfuse.yml`) through a
> `RedactingSink` that scrubs secrets/PII before export, while the local JSONL
> store stays exact for replay — a service run reconstructs by `run_id` end to end.
> The fan-out is a no-op when Langfuse isn't configured. *Remaining: stand up the
> Langfuse/Prometheus stack and wire Alertmanager — deployment infra.*

- Self-hosted Langfuse (Docker) as the `TraceSink` fanout target; retention + PII-safe trace redaction.
- Metrics (Prometheus) + dashboards + alerting on the eval/security/latency/cost gates (a risen attack-success-rate pages someone).
- Deterministic replay wired to the hosted store; `aegismem replay` works against production runs.
- **Gate:** a production run appears as a Langfuse waterfall and replays by `run_id`; alerts fire on injected gate breaches in staging.

## Phase P8 — Deploy, CI/CD, release ✅ done
**~3–5 days.** One command ships it; regressions block the ship.

> **Landed:** a multi-stage `Dockerfile` (uv build → slim non-root runtime, image
> healthcheck, `aegismem serve` entrypoint) with a `.dockerignore`; a full-stack
> `ops/docker-compose.yml` (service + pgvector, keys injected at runtime). A
> `deploy` workflow gated on the `ci` workflow succeeding: build+push
> `ghcr.io/<repo>:<sha>` → deploy staging → `ops/smoke.py` → promote to prod
> behind a manual `production` environment approval — so a dropped
> eval/security/latency gate blocks the image, and a failed smoke blocks
> promotion. A `serve` CLI runs uvicorn; `docs/RUNBOOK.md` covers migrations,
> SHA-tag rollback, and alert response. Alembic migrations are scaffolded in
> `migrations/` (initial revision applies the pgvector schema). *Remaining: point
> the workflow's deploy/promote steps at your actual platform.*

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
