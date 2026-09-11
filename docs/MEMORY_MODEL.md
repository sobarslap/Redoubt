# MEMORY MODEL

The memory subsystem (`src/aegismem/memory/`) is the runtime's source of truth. A
memory is never a bare `{"fact": "..."}` — every record is a typed, auditable,
provenance-carrying object with a governed lifecycle.

## The record

`MemoryRecord` (`models.py`) carries: `id`, `type` (tier), `content`, `status`
(lifecycle), `confidence`, `source`, `trust` (provenance label), `created_at`,
`updated_at`, `version`, and the provenance edges `supersedes` / `derived_from`.

### Tiers (`MemoryType`)
`WORKING` (run-local scratch) · `SESSION` (sliding conversation) · `SEMANTIC`
(durable facts) · `EPISODIC` (events, esp. failures) · `PROCEDURAL` (SOPs).

## Lifecycle state machine

Every status change is a governed transition (`lifecycle.py`, enforced by
`assert_transition`), and each writes an auditable `MemoryTransition` row
(reason / source / trigger / prev-version / timestamp):

```
CANDIDATE → VALIDATING → ACTIVE → SUPERSEDED → ARCHIVED → DELETED
```

Illegal edges are refused. This governed path is what upholds
**memory_corruption = 0**: nothing reaches `ACTIVE` truth except through an
allowed transition.

## Deterministic conflict resolution (LLM assists, never decides)

`ConflictResolver` (`conflict.py`) ingests a candidate against the live set:

```
new fact → exact-duplicate check (deterministic) → classify vs each live memory
        → verdict {relation, confidence} → deterministic policy layer
        → COMMIT_NEW | COMMIT_SUPERSEDE | REJECT_DUPLICATE | HUMAN_REVIEW
```

The classifier (LLM in production, `HeuristicConflictClassifier` offline) only
*proposes* a verdict; a fixed threshold policy maps `(relation, confidence)` to a
state-machine action. Below the review threshold nothing is mutated —
low-confidence conflicts go to a `HUMAN_REVIEW` queue rather than silently
rewriting memory. A supersede sets the new record `ACTIVE`, transitions the old to
`SUPERSEDED`, and records a `supersedes` provenance edge.

## Provenance graph

Edges (`provenance.py`, `EdgeKind`): `supersedes`, `derived_from`, `supported_by`.
`store.provenance(id)` returns the neighbourhood (what this replaced, what it was
derived from, what supports/depends on it) — answering *why does the agent believe
this?* Exposed over the service at `/memory/{id}/provenance`.

## Persistence & the swap path

`MemoryStore` is a `Protocol`, not a base class. `SQLiteMemoryStore` (WAL mode) is
the single-node dev/demo tier and the authoritative source of truth; the vector
and lexical indexes are derived, rebuildable accelerators. `PgVectorMemoryStore`
(Production P2) satisfies the same contract row-for-row for the networked
production tier — proven by the parity test. `TenantScopedMemoryStore` (P4) wraps
either backend to enforce per-tenant isolation at the store boundary.

## Tests

`tests/unit/test_memory_store.py` (CRUD + persistence) and
`test_memory_semantics.py` (dedup / conflict / supersede / lifecycle / provenance);
cross-tenant isolation in `tests/integration/test_auth_tenancy.py`; SQLite↔pgvector
parity in `tests/integration/test_pgvector_parity.py`.
