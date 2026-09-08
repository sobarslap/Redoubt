"""Phase 1 gate: MemoryStore CRUD against an in-memory SQLite store."""

from __future__ import annotations

import pytest

from aegismem.errors import NotFoundError
from aegismem.memory.models import (
    MemoryCreate,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    TrustLevel,
)
from aegismem.memory.store import MemoryStore, SQLiteMemoryStore


@pytest.fixture
def store() -> SQLiteMemoryStore:
    s = SQLiteMemoryStore(":memory:")
    yield s
    s.close()


def test_sqlite_store_satisfies_protocol(store: SQLiteMemoryStore) -> None:
    assert isinstance(store, MemoryStore)


def test_create_and_get_roundtrip(store: SQLiteMemoryStore) -> None:
    created = store.create(
        MemoryCreate(
            type=MemoryType.SEMANTIC,
            content="primary DB is postgres",
            confidence=0.94,
            source="tool:metrics_api",
            trust=TrustLevel.TOOL_RESULT,
        )
    )
    assert created.id.startswith("mem_")
    assert created.version == 1

    fetched = store.get(created.id)
    assert fetched == created
    assert fetched.type is MemoryType.SEMANTIC
    assert fetched.trust is TrustLevel.TOOL_RESULT


def test_get_missing_raises(store: SQLiteMemoryStore) -> None:
    with pytest.raises(NotFoundError):
        store.get("mem_does_not_exist")


def test_update_bumps_version_and_touches(store: SQLiteMemoryStore) -> None:
    rec = store.create(MemoryCreate(type=MemoryType.SESSION, content="draft"))
    updated = store.update(rec.id, MemoryUpdate(content="final", status=MemoryStatus.ACTIVE))
    assert updated.version == 2
    assert updated.content == "final"
    assert updated.status is MemoryStatus.ACTIVE
    assert updated.updated_at >= rec.updated_at


def test_empty_update_is_noop(store: SQLiteMemoryStore) -> None:
    rec = store.create(MemoryCreate(type=MemoryType.WORKING, content="x"))
    same = store.update(rec.id, MemoryUpdate())
    assert same.version == 1


def test_soft_delete_marks_deleted(store: SQLiteMemoryStore) -> None:
    rec = store.create(MemoryCreate(type=MemoryType.EPISODIC, content="incident"))
    store.delete(rec.id)
    assert store.get(rec.id).status is MemoryStatus.DELETED


def test_hard_delete_removes_row(store: SQLiteMemoryStore) -> None:
    rec = store.create(MemoryCreate(type=MemoryType.PROCEDURAL, content="runbook"))
    store.delete(rec.id, hard=True)
    with pytest.raises(NotFoundError):
        store.get(rec.id)


def test_list_filters_by_type_and_status(store: SQLiteMemoryStore) -> None:
    store.create(MemoryCreate(type=MemoryType.SEMANTIC, content="a"))
    store.create(MemoryCreate(type=MemoryType.SEMANTIC, content="b"))
    store.create(MemoryCreate(type=MemoryType.EPISODIC, content="c"))

    semantic = store.list(type=MemoryType.SEMANTIC)
    assert len(semantic) == 2
    assert {m.content for m in semantic} == {"a", "b"}

    candidates = store.list(status=MemoryStatus.CANDIDATE)
    assert len(candidates) == 3


def test_persistence_across_reopen(tmp_path) -> None:
    db = tmp_path / "persist.db"
    with SQLiteMemoryStore(db) as s:
        rec = s.create(MemoryCreate(type=MemoryType.SEMANTIC, content="durable"))
    with SQLiteMemoryStore(db) as s2:
        assert s2.get(rec.id).content == "durable"
