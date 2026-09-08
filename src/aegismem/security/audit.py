"""Audit log — an append-only record of privileged actions.

Distinct from the observability trace (which records *why* a run behaved as it
did): the audit log records *who* did *what* — authenticated runs, auth failures,
cancellations, memory deletes — so a security review has a tamper-evident trail.
In-process by default; a deployment points ``sink`` at a durable/WORM store.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class AuditEntry:
    action: str  # e.g. "run.create", "auth.fail", "run.cancel", "memory.delete"
    principal_id: str  # "-" when unauthenticated
    tenant: str
    at: str
    detail: str = ""
    outcome: str = "ok"  # ok | denied


@dataclass
class AuditLog:
    entries: list[AuditEntry] = field(default_factory=list)
    sink: Callable[[AuditEntry], None] | None = None
    max_entries: int = 10_000

    def record(
        self,
        action: str,
        *,
        principal_id: str = "-",
        tenant: str = "-",
        detail: str = "",
        outcome: str = "ok",
    ) -> AuditEntry:
        entry = AuditEntry(
            action=action,
            principal_id=principal_id,
            tenant=tenant,
            at=datetime.now(UTC).isoformat(),
            detail=detail,
            outcome=outcome,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:  # bound in-process memory
            del self.entries[: len(self.entries) - self.max_entries]
        if self.sink is not None:
            with contextlib.suppress(Exception):  # auditing must never break the request path
                self.sink(entry)
        return entry
