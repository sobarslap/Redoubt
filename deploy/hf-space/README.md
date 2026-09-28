---
title: AegisMem
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AegisMem — a production-grade agent runtime (live demo)

A framework-free, pure-Python agent runtime: lifecycle-managed memory, hybrid
retrieval, verified context compaction, a structural prompt-injection trust
boundary, guarded tool execution, and deterministic replay.

This Space serves the project's FastAPI service with the DevOps/SRE incident
memories pre-seeded, so you can try the full grounded, cited behavior live.

## Try it

Open **`/docs`** (the interactive API explorer), expand **`POST /runs`**,
click *Try it out*, and send:

```json
{"session_id": "s", "input": "why is the checkout service erroring?"}
```

- A **grounded** question returns a cited answer over the seeded memories.
- An **ungrounded** question (e.g. "what is the capital of France?") is refused —
  no memory, no fabrication.

Endpoints: `/docs`, `/healthz`, `/readyz`, `/metrics`, `POST /runs`,
`GET /runs/{run_id}`, `GET /memory/{memory_id}/provenance`.

Source: <https://github.com/sobarslap/Redoubt>
