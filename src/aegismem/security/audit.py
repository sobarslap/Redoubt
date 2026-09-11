"""Audit log — an append-only record of privileged actions.

Distinct from the observability trace (which records *why* a run behaved as it
did): the audit log records *who* did *what* — authenticated runs, auth failures,
cancellations, memory deletes — so a security review has a tamper-evident trail.
In-process by default; a deployment points ``sink`` at a durable/WORM store.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


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


@dataclass
class JSONLAuditSink:
    """Durable append-only audit sink — one JSON line per entry, fsync'd.

    A file-backed default for deployments; append-only + fsync gives a
    tamper-evident-ish trail on ordinary disks, and the same interface fronts a
    WORM bucket or SIEM forwarder in a hardened environment. Wire it in:
    ``AuditLog(sink=JSONLAuditSink("audit/audit.jsonl"))``.
    """

    path: str | Path = "audit/audit.jsonl"

    def __call__(self, entry: AuditEntry) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
            fh.flush()
            import os

            os.fsync(fh.fileno())
