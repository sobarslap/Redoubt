"""Tenant-scoped memory — cross-tenant isolation enforced at the store boundary.

A `TenantScopedMemoryStore` wraps any `MemoryStore` and binds every operation to
one `tenant`. Ownership is recorded on create and checked on every read/mutate:
a caller can never `get`, `list`, mutate, or even confirm the existence of
another tenant's memory — a cross-tenant `get` raises `NotFoundError`, not a
permission error, so existence itself never leaks.

Ownership lives in a shared `TenantOwnership` map. In-process by default; a
production backend records the tenant as a column and filters in SQL (the same
contract, enforced in the query) — this wrapper is the enforcement *semantics*,
proven by the adversarial isolation test.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from typing import cast

from aegismem.errors import NotFoundError
from aegismem.memory.models import MemoryCreate, MemoryRecord, MemoryStatus, MemoryType


@dataclass
class TenantOwnership:
    """memory_id -> owning tenant. Shared across per-tenant store views."""

    _owner: dict[str, str] = field(default_factory=dict)

    def claim(self, memory_id: str, tenant: str) -> None:
        self._owner[memory_id] = tenant

    def owner(self, memory_id: str) -> str | None:
        return self._owner.get(memory_id)

    def owned_by(self, tenant: str) -> set[str]:
        return {mid for mid, t in self._owner.items() if t == tenant}


class TenantScopedMemoryStore:
    """A per-tenant view over a shared underlying `MemoryStore`."""

    def __init__(self, inner: object, tenant: str, ownership: TenantOwnership) -> None:
        self._inner = inner
        self._tenant = tenant
        self._own = ownership

    def _require_owned(self, memory_id: str, stage: str) -> None:
        if self._own.owner(memory_id) != self._tenant:
            # Never reveal that the id exists for another tenant.
            raise NotFoundError(f"memory {memory_id!r} not found", stage=stage)

    # -- CRUD ------------------------------------------------------------------

    def create(self, data: MemoryCreate) -> MemoryRecord:
        rec = cast(MemoryRecord, self._inner.create(data))  # type: ignore[attr-defined]
        self._own.claim(rec.id, self._tenant)
        return rec

    def get(self, memory_id: str) -> MemoryRecord:
        self._require_owned(memory_id, "memory.get")
        return cast(MemoryRecord, self._inner.get(memory_id))  # type: ignore[attr-defined]

    def update(self, memory_id: str, changes: object) -> MemoryRecord:
        self._require_owned(memory_id, "memory.update")
        return cast(MemoryRecord, self._inner.update(memory_id, changes))  # type: ignore[attr-defined]

    def delete(self, memory_id: str, *, hard: bool = False) -> None:
        self._require_owned(memory_id, "memory.delete")
        self._inner.delete(memory_id, hard=hard)  # type: ignore[attr-defined]

    def list(
        self,
        *,
        type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[MemoryRecord]:
        owned = self._own.owned_by(self._tenant)
        rows = self._inner.list(  # type: ignore[attr-defined]
            type=type, status=status, limit=10_000, offset=0
        )
        scoped = [r for r in rows if r.id in owned]
        return scoped[offset : offset + limit]

    # -- lifecycle / provenance (ownership-checked passthrough) ---------------

    def transition(self, memory_id: str, to_status: MemoryStatus, **kw: object) -> MemoryRecord:
        self._require_owned(memory_id, "memory.transition")
        return cast(
            MemoryRecord,
            self._inner.transition(memory_id, to_status, **kw),  # type: ignore[attr-defined]
        )

    def add_edge(self, src_id: str, dst_id: str, kind: str) -> None:
        self._require_owned(src_id, "memory.add_edge")
        self._require_owned(dst_id, "memory.add_edge")
        self._inner.add_edge(src_id, dst_id, kind)  # type: ignore[attr-defined]

    def provenance(self, memory_id: str) -> object:
        self._require_owned(memory_id, "memory.provenance")
        return self._inner.provenance(memory_id)  # type: ignore[attr-defined]
