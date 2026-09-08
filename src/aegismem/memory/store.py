"""MemoryStore: the source-of-truth persistence abstraction.

SQLite (WAL mode) is authoritative. The vector and lexical indexes added in
Phase 3 are derived, rebuildable accelerators; this interface owns the truth.

The abstraction is a ``Protocol`` so alternative backends (pgvector / Qdrant, per
the documented production swap path) can satisfy the same contract without
inheriting from a base class.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from aegismem.errors import NotFoundError, StorageError
from aegismem.memory.models import (
    MemoryCreate,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    TrustLevel,
)


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
