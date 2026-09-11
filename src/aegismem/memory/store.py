"""MemoryStore: the source-of-truth persistence abstraction.

SQLite (WAL mode) is authoritative. The vector and lexical indexes added in
Phase 3 are derived, rebuildable accelerators; this interface owns the truth.

The abstraction is a ``Protocol`` so alternative backends (pgvector / Qdrant, per
the documented production swap path) can satisfy the same contract without
inheriting from a base class.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from aegismem.errors import NotFoundError, StorageError
from aegismem.memory.conflict import ConflictVerdict
from aegismem.memory.lifecycle import MemoryTransition, assert_transition
from aegismem.memory.models import (
    MemoryCreate,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    TrustLevel,
)
from aegismem.memory.provenance import EdgeKind, ProvenanceView


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


@runtime_checkable
class MemoryStore(Protocol):
    """CRUD contract for memory persistence."""

    def create(self, data: MemoryCreate) -> MemoryRecord: ...
    def get(self, memory_id: str) -> MemoryRecord: ...
    def update(self, memory_id: str, changes: object) -> MemoryRecord: ...
    def delete(self, memory_id: str, *, hard: bool = False) -> None: ...
    def list(
        self,
        *,
        type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    content      TEXT NOT NULL,
    status       TEXT NOT NULL,
    confidence   REAL NOT NULL,
    source       TEXT NOT NULL,
    trust        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    version      INTEGER NOT NULL,
    supersedes   TEXT,
    derived_from TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE TABLE IF NOT EXISTS transitions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id    TEXT NOT NULL,
    from_status  TEXT,
    to_status    TEXT NOT NULL,
    reason       TEXT NOT NULL,
    source       TEXT NOT NULL,
    trigger      TEXT NOT NULL,
    prev_version INTEGER NOT NULL,
    at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transitions_mem ON transitions(memory_id);

CREATE TABLE IF NOT EXISTS provenance_edges (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind   TEXT NOT NULL,
    at     TEXT NOT NULL,
    UNIQUE(src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_prov_src ON provenance_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_prov_dst ON provenance_edges(dst_id);

CREATE TABLE IF NOT EXISTS review_queue (
    id            TEXT PRIMARY KEY,
    candidate_json TEXT NOT NULL,
    existing_id   TEXT,
    relation      TEXT NOT NULL,
    confidence    REAL NOT NULL,
    rationale     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
"""


class SQLiteMemoryStore:
    """SQLite-backed :class:`MemoryStore`.

    Opens the database in WAL mode for concurrent reads and durable writes. Pass
    ``":memory:"`` for an ephemeral store (tests).
    """

    def __init__(self, db_path: str | Path = "aegismem.db") -> None:
        self._path = str(db_path)
        # check_same_thread=False keeps the async API layer simple; access is
        # serialized by SQLite's own locking plus WAL.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- serialization helpers -------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            status=MemoryStatus(row["status"]),
            confidence=row["confidence"],
            source=row["source"],
            trust=TrustLevel(row["trust"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
            supersedes=row["supersedes"],
            derived_from=json.loads(row["derived_from"]),
        )

    @staticmethod
    def _record_params(rec: MemoryRecord) -> tuple[object, ...]:
        return (
            rec.id,
            rec.type.value,
            rec.content,
            rec.status.value,
            rec.confidence,
            rec.source,
            rec.trust.value,
            rec.created_at.isoformat(),
            rec.updated_at.isoformat(),
            rec.version,
            rec.supersedes,
            json.dumps(rec.derived_from),
        )

    # -- CRUD ------------------------------------------------------------------

    def create(self, data: MemoryCreate) -> MemoryRecord:
        record = MemoryRecord(**data.model_dump())
        try:
            self._conn.execute(
                """INSERT INTO memories
                   (id, type, content, status, confidence, source, trust,
                    created_at, updated_at, version, supersedes, derived_from)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                self._record_params(record),
            )
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise StorageError(f"failed to create memory: {exc}", stage="memory.create") from exc
        return record

    def get(self, memory_id: str) -> MemoryRecord:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"memory {memory_id!r} not found", stage="memory.get")
        return self._row_to_record(row)

    def update(self, memory_id: str, changes: object) -> MemoryRecord:
        from aegismem.memory.models import MemoryUpdate

        if not isinstance(changes, MemoryUpdate):
            raise TypeError("changes must be a MemoryUpdate")
        record = self.get(memory_id)
        patch = changes.model_dump(exclude_unset=True)
        if not patch:
            return record
        updated = record.model_copy(update=patch)
        updated.version = record.version + 1
        updated.touch()
        try:
            self._conn.execute(
                """UPDATE memories SET
                    type=?, content=?, status=?, confidence=?, source=?, trust=?,
                    created_at=?, updated_at=?, version=?, supersedes=?, derived_from=?
                   WHERE id=?""",
                (*self._record_params(updated)[1:], updated.id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise StorageError(f"failed to update memory: {exc}", stage="memory.update") from exc
        return updated

    def delete(self, memory_id: str, *, hard: bool = False) -> None:
        """Soft-delete by default (status -> DELETED); ``hard`` removes the row."""
        record = self.get(memory_id)  # raises NotFoundError if absent
        if hard:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
            return
        record.status = MemoryStatus.DELETED
        record.version += 1
        record.touch()
        self._conn.execute(
            "UPDATE memories SET status=?, version=?, updated_at=? WHERE id=?",
            (record.status.value, record.version, record.updated_at.isoformat(), memory_id),
        )
        self._conn.commit()

    def list(
        self,
        *,
        type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = self._conn.execute(
            f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # -- lifecycle (Phase 2) ---------------------------------------------------

    def transition(
        self,
        memory_id: str,
        to_status: MemoryStatus,
        *,
        reason: str,
        source: str = "system",
        trigger: str = "manual",
    ) -> MemoryRecord:
        """Move a memory along an allowed lifecycle edge, writing an audit row.

        Refuses illegal edges via :func:`assert_transition` — the governed path
        that upholds ``memory_corruption = 0``.
        """
        record = self.get(memory_id)
        assert_transition(record.status, to_status)
        from_status = record.status
        prev_version = record.version
        record.status = to_status
        record.version += 1
        record.touch()
        try:
            self._conn.execute(
                "UPDATE memories SET status=?, version=?, updated_at=? WHERE id=?",
                (record.status.value, record.version, record.updated_at.isoformat(), memory_id),
            )
            self._conn.execute(
                """INSERT INTO transitions
                   (memory_id, from_status, to_status, reason, source, trigger, prev_version, at)
                   VALUES (?,?,?,?,?,?,?,?)""",
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
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise StorageError(
                f"failed to transition memory: {exc}", stage="memory.transition"
            ) from exc
        return record

    def history(self, memory_id: str) -> builtins.list[MemoryTransition]:
        rows = self._conn.execute(
            "SELECT * FROM transitions WHERE memory_id = ? ORDER BY id ASC", (memory_id,)
        ).fetchall()
        return [
            MemoryTransition(
                memory_id=r["memory_id"],
                from_status=MemoryStatus(r["from_status"]) if r["from_status"] else None,
                to_status=MemoryStatus(r["to_status"]),
                reason=r["reason"],
                source=r["source"],
                trigger=r["trigger"],
                prev_version=r["prev_version"],
                at=datetime.fromisoformat(r["at"]),
            )
            for r in rows
        ]

    # -- provenance graph (Phase 2) --------------------------------------------

    def add_edge(self, src_id: str, dst_id: str, kind: str) -> None:
        """Insert a provenance edge ``src -> dst`` (idempotent)."""
        EdgeKind(kind)  # validate vocabulary
        self._conn.execute(
            """INSERT OR IGNORE INTO provenance_edges (src_id, dst_id, kind, at)
               VALUES (?,?,?,?)""",
            (src_id, dst_id, kind, _utcnow_iso()),
        )
        self._conn.commit()

    def provenance(self, memory_id: str) -> ProvenanceView:
        """Return the provenance neighbourhood of ``memory_id``."""
        self.get(memory_id)  # raises NotFoundError if absent
        out = self._conn.execute(
            "SELECT dst_id, kind FROM provenance_edges WHERE src_id = ?", (memory_id,)
        ).fetchall()
        inc = self._conn.execute(
            "SELECT src_id, kind FROM provenance_edges WHERE dst_id = ?", (memory_id,)
        ).fetchall()
        view = ProvenanceView(memory_id=memory_id)
        for r in out:
            if r["kind"] == EdgeKind.SUPERSEDES:
                view.supersedes.append(r["dst_id"])
            elif r["kind"] == EdgeKind.DERIVED_FROM:
                view.derived_from.append(r["dst_id"])
            elif r["kind"] == EdgeKind.SUPPORTED_BY:
                view.supported_by.append(r["dst_id"])
        for r in inc:
            if r["kind"] == EdgeKind.SUPERSEDES:
                view.superseded_by.append(r["src_id"])
            elif r["kind"] == EdgeKind.SUPPORTED_BY:
                view.supports.append(r["src_id"])
        return view

    # -- human-review queue (Phase 2) ------------------------------------------

    def enqueue_review(
        self, candidate: MemoryCreate, existing_id: str, verdict: ConflictVerdict
    ) -> str:
        review_id = f"rev_{uuid.uuid4().hex[:16]}"
        self._conn.execute(
            """INSERT INTO review_queue
               (id, candidate_json, existing_id, relation, confidence, rationale, status, at)
               VALUES (?,?,?,?,?,?,?,?)""",
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
        self._conn.commit()
        return review_id

    def review_queue(self, status: str = "pending") -> builtins.list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY at ASC", (status,)
        ).fetchall()
        return [dict(r) for r in rows]
