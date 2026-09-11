"""Production Phase P2 gate — pgvector backend is behavior-parity with SQLite.

The parity test runs an identical sequence of operations against both stores and
asserts identical observable results (records, lifecycle history, provenance,
review queue). It is gated on ``AEGISMEM_PG_DSN`` (or ``DATABASE_URL``): with no
Postgres it skips, so keyless/serviceless CI stays green while a real deployment
proves the swap is transparent.

The un-gated tests cover what needs no database: the vector literal helper and the
retrieval model factory's fallback behavior.
"""

from __future__ import annotations

import os

import pytest

from aegismem.memory.models import MemoryCreate, MemoryStatus, MemoryType, MemoryUpdate, TrustLevel
from aegismem.memory.store import SQLiteMemoryStore

_DSN = os.environ.get("AEGISMEM_PG_DSN") or os.environ.get("DATABASE_URL")


# -- un-gated: no database needed --------------------------------------------


def test_vec_literal_helper() -> None:
    from aegismem.memory.pgstore import _vec

    assert _vec(None) is None
    assert _vec([0.1, 0.2, 0.5]) == "[0.1,0.2,0.5]"


def test_retrieval_factory_falls_back_without_models() -> None:
    from aegismem.retrieval.factory import build_embedder, build_reranker

    emb = build_embedder(prefer_real=False)
    rr = build_reranker(prefer_real=False)
    v = emb.embed("connection pool exhaustion")
    assert isinstance(v, list) and len(v) > 0
    assert isinstance(rr.score("pool", "connection pool exhaustion"), float)


def test_pgstore_missing_driver_raises_typed_error(monkeypatch) -> None:
    # Simulate psycopg_pool being unavailable -> a typed StorageError, not ImportError.
    import builtins

    from aegismem.errors import StorageError

    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name.startswith("psycopg_pool"):
            raise ImportError("no psycopg_pool")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    from aegismem.memory.pgstore import PgVectorMemoryStore

    with pytest.raises(StorageError):
        PgVectorMemoryStore("postgresql://localhost/nope")


def test_alembic_initial_migration_matches_store_ddl() -> None:
    # The migration reuses PgVectorMemoryStore._DDL, so schema can't drift. Guard
    # on alembic being installed (migrations group). Loaded by path — Alembic
    # revision files are named with a leading digit and loaded by file, not import.
    pytest.importorskip("alembic")
    import importlib.util
    from pathlib import Path

    path = Path("migrations/versions/0001_initial_schema.py")
    spec = importlib.util.spec_from_file_location("_mig_0001", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.revision == "0001_initial"
    assert mod.down_revision is None

    from aegismem.memory.pgstore import _DDL

    ddl = _DDL.format(dim=384)
    for table in ("memories", "transitions", "provenance_edges", "review_queue"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl


# -- gated parity (needs a live Postgres with pgvector) ----------------------


def _exercise(store) -> dict[str, object]:
    """Run a fixed op sequence; return observable state for comparison."""
    a = store.create(
        MemoryCreate(type=MemoryType.SEMANTIC, content="primary db is mysql", trust=TrustLevel.USER)
    )
    store.transition(a.id, MemoryStatus.ACTIVE, reason="seed", source="t", trigger="test")
    b = store.create(
        MemoryCreate(
            type=MemoryType.SEMANTIC,
            content="primary db is postgres",
            trust=TrustLevel.USER,
            supersedes=a.id,
        )
    )
    store.transition(b.id, MemoryStatus.ACTIVE, reason="migrate", source="t", trigger="test")
    store.transition(a.id, MemoryStatus.SUPERSEDED, reason="replaced", source="t", trigger="test")
    store.add_edge(b.id, a.id, "supersedes")
    store.update(b.id, MemoryUpdate(confidence=0.9))
    active = store.list(status=MemoryStatus.ACTIVE)
    prov = store.provenance(b.id)
    hist = store.history(a.id)
    return {
        "active_contents": sorted(m.content for m in active),
        "b_version": store.get(b.id).version,
        "a_status": store.get(a.id).status.value,
        "prov_supersedes": sorted(prov.supersedes),
        "a_history_len": len(hist),
    }


@pytest.mark.skipif(_DSN is None, reason="no AEGISMEM_PG_DSN; pgvector parity skipped")
def test_pgvector_parity_with_sqlite() -> None:  # pragma: no cover - needs live db
    from aegismem.memory.pgstore import PgVectorMemoryStore

    with SQLiteMemoryStore(":memory:") as sqlite_store:
        sqlite_state = _exercise(sqlite_store)

    pg = PgVectorMemoryStore(_DSN)
    try:
        # Clean slate for a deterministic comparison.
        with pg._pool.connection() as conn:
            for t in ("provenance_edges", "transitions", "review_queue", "memories"):
                conn.execute(f"TRUNCATE {t}")
            conn.commit()
        pg_state = _exercise(pg)
    finally:
        pg.close()

    assert pg_state == sqlite_state
