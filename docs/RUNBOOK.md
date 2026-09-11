# OPERATIONS RUNBOOK (Production Phase P8)

Deploy, migrate, roll back, and respond to alerts for the AegisMem service. The
demo's own domain — DevOps/SRE incident response — dogfooded on the service itself.

## Build & run

```bash
docker build -t aegismem:local .
docker compose -f ops/docker-compose.yml up --build        # service + pgvector
docker compose -f ops/docker-compose.langfuse.yml up -d     # optional trace backend
```

The image runs as non-root, `aegismem serve` on port 8000, with a `/healthz`
container healthcheck. Provider keys and `AEGISMEM_PG_DSN` are injected as env at
runtime — never baked into the image (enforced by the P4 secret scan).

## Release

Tag `main`; CI must be green (a dropped eval/security/latency/cost gate blocks the
build). The `deploy` workflow builds a `ghcr.io/<repo>:<sha>` image, deploys to
staging, runs `python -m ops.smoke <staging-url>`, and promotes to prod only after
the smoke test passes and a human approves the `production` GitHub Environment.

## Database migrations

SQLite (dev) and pgvector (prod) share the `MemoryStore` contract. For prod schema
changes: add an Alembic revision, apply with `alembic upgrade head` against a
backed-up database, and verify with the pgvector parity test
(`AEGISMEM_PG_DSN=... uv run pytest tests/integration/test_pgvector_parity.py`).
The derived vector/BM25 indexes are rebuildable from the source-of-truth rows.

## Rollback

1. Re-point the platform to the previous known-good image tag `ghcr.io/<repo>:<prev-sha>`.
2. If a migration shipped, reverse it on the **serving** database — take a fresh
   backup first, confirm the migration is reversible, then `alembic downgrade -1`
   against the live DSN. Never downgrade the backup: that leaves the serving
   database on the new schema and destroys your clean recovery point. If the
   migration is not safely reversible, restore the pre-deploy backup instead.
3. Confirm recovery with `python -m ops.smoke <url>`.
Images are immutable and versioned by commit SHA, so rollback is a tag swap.

## Alert response (from `ops/alerts.yml`)

| Alert | First action |
|---|---|
| `HighRunFailureRate` | Check `/readyz` in-flight + recent `run.*` traces; is a provider/DB down? |
| `RunLatencyP95TooHigh` | Check DB latency + LLM provider latency; scale replicas or shed load. |
| `GuardrailBlockSpike` | Likely an attack; inspect audit log for the source principal/tenant. |
| `UnauthorizedToolAttempts` | Injection or compromised caller; rotate the caller's key, review audit. |

## Debugging a specific run

Every run is replayable by `run_id` from the trace store:

```bash
uv run aegismem replay <run_id> --trace-path traces/service.jsonl
```

This reconstructs the ordered pipeline (guardrails -> retrieval -> execution ->
output) with each stage's decision attributes — the fastest path from "this run
misbehaved" to "here is exactly why."
