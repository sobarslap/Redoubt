"""Memory lifecycle state machine.

Transitions are governed, not free-form: a memory moves only along an allowed
edge, and every move is recorded as an auditable :class:`MemoryTransition`
carrying reason, source, trigger, and the version it left behind. This is the
spine of the ``memory_corruption = 0`` invariant — status changes go through
:func:`assert_transition` rather than arbitrary writes.

``CANDIDATE -> VALIDATING -> ACTIVE -> SUPERSEDED -> ARCHIVED -> DELETED``
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from aegismem.errors import AegisError, ErrorCategory
from aegismem.memory.models import MemoryStatus

# The allowed forward (and a few corrective) edges. Anything not listed is refused.
ALLOWED_TRANSITIONS: dict[MemoryStatus, frozenset[MemoryStatus]] = {
    MemoryStatus.CANDIDATE: frozenset(
        {MemoryStatus.VALIDATING, MemoryStatus.ACTIVE, MemoryStatus.DELETED}
    ),
    MemoryStatus.VALIDATING: frozenset(
        {MemoryStatus.ACTIVE, MemoryStatus.CANDIDATE, MemoryStatus.DELETED}
    ),
    MemoryStatus.ACTIVE: frozenset(
        {MemoryStatus.SUPERSEDED, MemoryStatus.ARCHIVED, MemoryStatus.DELETED}
    ),
    MemoryStatus.SUPERSEDED: frozenset({MemoryStatus.ARCHIVED, MemoryStatus.DELETED}),
    MemoryStatus.ARCHIVED: frozenset({MemoryStatus.DELETED}),
    MemoryStatus.DELETED: frozenset(),
}

# Statuses that count as "live" for retrieval / conflict candidacy.
LIVE_STATUSES: frozenset[MemoryStatus] = frozenset(
    {MemoryStatus.CANDIDATE, MemoryStatus.VALIDATING, MemoryStatus.ACTIVE}
)


class TransitionError(AegisError):
    """Raised when a status change is not an allowed lifecycle edge."""

    def __init__(self, frm: MemoryStatus, to: MemoryStatus) -> None:
        super().__init__(
            "invalid_transition",
            ErrorCategory.CONFLICT,
            f"illegal lifecycle transition {frm.value} -> {to.value}",
            stage="memory.transition",
        )


class MemoryTransition(BaseModel):
    """An audit record of one lifecycle move."""

    memory_id: str
    from_status: MemoryStatus | None
    to_status: MemoryStatus
    reason: str
    source: str = "system"
    trigger: str = "manual"
    prev_version: int
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def is_valid_transition(frm: MemoryStatus, to: MemoryStatus) -> bool:
    return to in ALLOWED_TRANSITIONS.get(frm, frozenset())


def assert_transition(frm: MemoryStatus, to: MemoryStatus) -> None:
    """Raise :class:`TransitionError` unless ``frm -> to`` is allowed."""
    if not is_valid_transition(frm, to):
        raise TransitionError(frm, to)
