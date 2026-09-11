"""PgVectorMemoryStore — the production `MemoryStore` backend (Production P2).

Postgres + pgvector behind the *exact same* `MemoryStore` contract as
`SQLiteMemoryStore`, so the runtime swaps backends with no code change. SQLite is
the single-node dev/demo tier; this is the networked, multi-writer production
tier. The embedding column co-locates the derived vector accelerator with the
source of truth, so KNN retrieval runs in the database.

Import-guarded: `psycopg` (v3) and `pgvector` live in a dependency group, so a
clean checkout and keyless CI never import them. Construction needs a DSN
(`AEGISMEM_PG_DSN` / `DATABASE_URL`); without one the runtime uses SQLite.

Semantics mirror `SQLiteMemoryStore` row-for-row: same lifecycle governance
(`assert_transition`), same provenance edges, same review queue, same typed
errors — verified by the parity test in `tests/integration/test_pgvector_parity.py`.
"""

from __future__ import annotations

import builtins
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aegismem.errors import NotFoundError, StorageError
from aegismem.memory.conflict import ConflictVerdict
from aegismem.memory.lifecycle import MemoryTransition, assert_transition
from aegismem.memory.models import (
    MemoryCreate,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    TrustLevel,
)
from aegismem.memory.provenance import EdgeKind, ProvenanceView
from aegismem.memory.store import _utcnow_iso

if TYPE_CHECKING:
    from aegismem.retrieval.embeddings import Embedder

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    content      TEXT NOT NULL,
    status       TEXT NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL,
    source       TEXT NOT NULL,
    trust        TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL,
    version      INTEGER NOT NULL,
    supersedes   TEXT,
    derived_from JSONB NOT NULL DEFAULT '[]',
    embedding    vector({dim})
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE TABLE IF NOT EXISTS transitions (
    id           BIGSERIAL PRIMARY KEY,
    memory_id    TEXT NOT NULL,
    from_status  TEXT,
    to_status    TEXT NOT NULL,
    reason       TEXT NOT NULL,
    source       TEXT NOT NULL,
    trigger      TEXT NOT NULL,
    prev_version INTEGER NOT NULL,
    at           TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transitions_mem ON transitions(memory_id);

CREATE TABLE IF NOT EXISTS provenance_edges (
    id     BIGSERIAL PRIMARY KEY,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind   TEXT NOT NULL,
    at     TIMESTAMPTZ NOT NULL,
    UNIQUE(src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_prov_src ON provenance_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_prov_dst ON provenance_edges(dst_id);

CREATE TABLE IF NOT EXISTS review_queue (
    id             TEXT PRIMARY KEY,
    candidate_json TEXT NOT NULL,
    existing_id    TEXT,
    relation       TEXT NOT NULL,
    confidence     DOUBLE PRECISION NOT NULL,
    rationale      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    at             TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
"""


class PgVectorMemoryStore:
    """Postgres + pgvector `MemoryStore`. Uses a psycopg connection pool.

    ``embedder`` is optional; when provided, new/updated memories are embedded and
    `similar()` runs a KNN query in the database. Without it the store is a pure
    source-of-truth backend and the retrieval router owns the vector accelerator.
    """

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = 384,  # bge-small-en-v1.5
        embedder: Embedder | None = None,
        min_size: int = 1,
        max_size: int = 8,
    ) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except Exception as exc:  # pragma: no cover - optional dependency
            raise StorageError(
                f"psycopg/psycopg_pool not installed (uv sync --group pg): {exc}",
                retryable=False,
                stage="memory.pg",
            ) from exc
        self._dim = dim
        self._embedder = embedder
        self._pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)
        with self._pool.connection() as conn:
            conn.execute(_DDL.format(dim=dim))
            conn.commit()

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> PgVectorMemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> MemoryRecord:
        (
            mid,
            mtype,
            content,
            status,
            confidence,
            source,
            trust,
            created_at,
            updated_at,
            version,
            supersedes,
            derived_from,
        ) = row
        df = derived_from if isinstance(derived_from, list) else json.loads(derived_from or "[]")
        return MemoryRecord(
            id=str(mid),
            type=MemoryType(mtype),
            content=str(content),
            status=MemoryStatus(status),
            confidence=float(confidence),
            source=str(source),
            trust=TrustLevel(trust),
            created_at=created_at
            if isinstance(created_at, datetime)
            else datetime.fromisoformat(str(created_at)),
            updated_at=updated_at
            if isinstance(updated_at, datetime)
            else datetime.fromisoformat(str(updated_at)),
            version=int(version),
            supersedes=supersedes,
            derived_from=df,
        )

    _COLS = (
        "id, type, content, status, confidence, source, trust, "
        "created_at, updated_at, version, supersedes, derived_from"
    )

    def _embed(self, text: str) -> list[float] | None:
        return self._embedder.embed(text) if self._embedder is not None else None

    # -- CRUD ------------------------------------------------------------------

    def create(self, data: MemoryCreate) -> MemoryRecord:
        record = MemoryRecord(**data.model_dump())
        emb = self._embed(record.content)
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    f"INSERT INTO memories ({self._COLS}, embedding) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        record.id,
                        record.type.value,
                        record.content,
                        record.status.value,
                        record.confidence,
                        record.source,
                        record.trust.value,
                        record.created_at,
                        record.updated_at,
                        record.version,
                        record.supersedes,
                        json.dumps(record.derived_from),
                        _vec(emb),
                    ),
                )
                conn.commit()
        except Exception as exc:  # pragma: no cover - needs live db
            raise StorageError(f"failed to create memory: {exc}", stage="memory.create") from exc
        return record

    def get(self, memory_id: str) -> MemoryRecord:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {self._COLS} FROM memories WHERE id = %s", (memory_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"memory {memory_id!r} not found", stage="memory.get")
        return self._row_to_record(row)

    def update(self, memory_id: str, changes: object) -> MemoryRecord:
        if not isinstance(changes, MemoryUpdate):
            raise TypeError("changes must be a MemoryUpdate")
        record = self.get(memory_id)
        patch = changes.model_dump(exclude_unset=True)
        if not patch:
            return record
        updated = record.model_copy(update=patch)
        updated.version = record.version + 1
        updated.touch()
        emb = self._embed(updated.content)
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "UPDATE memories SET type=%s, content=%s, status=%s, confidence=%s, "
                    "source=%s, trust=%s, created_at=%s, updated_at=%s, version=%s, "
                    "supersedes=%s, derived_from=%s, embedding=%s WHERE id=%s",
                    (
                        updated.type.value,
                        updated.content,
                        updated.status.value,
                        updated.confidence,
                        updated.source,
                        updated.trust.value,
                        updated.created_at,
                        updated.updated_at,
                        updated.version,
                        updated.supersedes,
                        json.dumps(updated.derived_from),
                        _vec(emb),
                        updated.id,
                    ),
                )
                conn.commit()
        except Exception as exc:  # pragma: no cover - needs live db
            raise StorageError(f"failed to update memory: {exc}", stage="memory.update") from exc
        return updated

    def delete(self, memory_id: str, *, hard: bool = False) -> None:
        record = self.get(memory_id)
        with self._pool.connection() as conn:
            if hard:
                conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
                conn.commit()
                return
            record.status = MemoryStatus.DELETED
            record.version += 1
            record.touch()
            conn.execute(
                "UPDATE memories SET status=%s, version=%s, updated_at=%s WHERE id=%s",
                (record.status.value, record.version, record.updated_at, memory_id),
            )
            conn.commit()

    def list(
        self,
        *,
        type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if type is not None:
            clauses.append("type = %s")
            params.append(type.value)
        if status is not None:
            clauses.append("status = %s")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM memories {where} "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                params,
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # -- lifecycle -------------------------------------------------------------

    def transition(
        self,
        memory_id: str,
        to_status: MemoryStatus,
        *,
        reason: str,
        source: str = "system",
        trigger: str = "manual",
    ) -> MemoryRecord:
        record = self.get(memory_id)
        assert_transition(record.status, to_status)
        from_status = record.status
        prev_version = record.version
        record.status = to_status
        record.version += 1
        record.touch()
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "UPDATE memories SET status=%s, version=%s, updated_at=%s WHERE id=%s",
                    (record.status.value, record.version, record.updated_at, memory_id),
                )
                conn.execute(
                    "INSERT INTO transitions (memory_id, from_status, to_status, reason, "
                    "source, trigger, prev_version, at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        memory_id,
                        from_status.value,
                        to_status.value,
                        reason,
                        source,
                        trigger,
                        prev_version,
                        _utcnow_iso(),
                    ),
                )
                conn.commit()
        except Exception as exc:  # pragma: no cover - needs live db
            if isinstance(exc, StorageError):
                raise
            raise StorageError(
                f"failed to transition memory: {exc}", stage="memory.transition"
            ) from exc
        return record

    def history(self, memory_id: str) -> builtins.list[MemoryTransition]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT memory_id, from_status, to_status, reason, source, trigger, "
                "prev_version, at FROM transitions WHERE memory_id = %s ORDER BY id ASC",
                (memory_id,),
            ).fetchall()
        return [
            MemoryTransition(
                memory_id=r[0],
                from_status=MemoryStatus(r[1]) if r[1] else None,
                to_status=MemoryStatus(r[2]),
                reason=r[3],
                source=r[4],
                trigger=r[5],
                prev_version=r[6],
                at=r[7] if isinstance(r[7], datetime) else datetime.fromisoformat(str(r[7])),
            )
            for r in rows
        ]

    # -- provenance ------------------------------------------------------------

    def add_edge(self, src_id: str, dst_id: str, kind: str) -> None:
        EdgeKind(kind)
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO provenance_edges (src_id, dst_id, kind, at) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (src_id, dst_id, kind) DO NOTHING",
                (src_id, dst_id, kind, _utcnow_iso()),
            )
            conn.commit()

    def provenance(self, memory_id: str) -> ProvenanceView:
        self.get(memory_id)
        with self._pool.connection() as conn:
            out = conn.execute(
                "SELECT dst_id, kind FROM provenance_edges WHERE src_id = %s", (memory_id,)
            ).fetchall()
            inc = conn.execute(
                "SELECT src_id, kind FROM provenance_edges WHERE dst_id = %s", (memory_id,)
            ).fetchall()
        view = ProvenanceView(memory_id=memory_id)
        for dst, kind in out:
            if kind == EdgeKind.SUPERSEDES:
                view.supersedes.append(dst)
            elif kind == EdgeKind.DERIVED_FROM:
                view.derived_from.append(dst)
            elif kind == EdgeKind.SUPPORTED_BY:
                view.supported_by.append(dst)
        for src, kind in inc:
            if kind == EdgeKind.SUPERSEDES:
                view.superseded_by.append(src)
            elif kind == EdgeKind.SUPPORTED_BY:
                view.supports.append(src)
        return view

    # -- review queue ----------------------------------------------------------

    def enqueue_review(
        self, candidate: MemoryCreate, existing_id: str, verdict: ConflictVerdict
    ) -> str:
        review_id = f"rev_{uuid.uuid4().hex[:16]}"
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO review_queue (id, candidate_json, existing_id, relation, "
                "confidence, rationale, status, at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    review_id,
                    candidate.model_dump_json(),
                    existing_id,
                    verdict.relation.value,
                    verdict.confidence,
                    verdict.rationale,
                    "pending",
                    _utcnow_iso(),
                ),
            )
            conn.commit()
        return review_id

    def review_queue(self, status: str = "pending") -> builtins.list[dict[str, object]]:
        with self._pool.connection() as conn:
            cur = conn.execute(
                "SELECT id, candidate_json, existing_id, relation, confidence, rationale, "
                "status, at FROM review_queue WHERE status = %s ORDER BY at ASC",
                (status,),
            )
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, r, strict=True)) for r in rows]

    # -- in-database KNN (the pgvector payoff) ---------------------------------

    def similar(self, query: str, *, k: int = 5) -> builtins.list[tuple[str, float]]:
        """KNN over the embedding column: returns ``(memory_id, distance)``.

        Requires an ``embedder``; the derived accelerator lives in the database
        rather than in-process."""
        if self._embedder is None:
            raise StorageError("similar() needs an embedder", retryable=False, stage="memory.pg")
        qv = _vec(self._embedder.embed(query))
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, embedding <=> %s AS dist FROM memories "
                "WHERE embedding IS NOT NULL AND status = %s ORDER BY dist ASC LIMIT %s",
                (qv, MemoryStatus.ACTIVE.value, k),
            ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]


def _vec(embedding: list[float] | None) -> str | None:
    """pgvector text literal, e.g. ``[0.1,0.2,...]`` — accepted by the vector type."""
    if embedding is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"
