"""Memory domain models: the rich memory record and its enumerations.

A memory is never a bare ``{"fact": "..."}``. Every record carries type (tier),
lifecycle status, confidence, provenance, and a trust label so downstream
subsystems (retrieval, conflict resolution, security) can reason about it.

Phase 1 defines the schema and persists it. The lifecycle *transitions* between
statuses (the state machine) are enforced in Phase 2; here the statuses exist and
records are created/read/updated/deleted as data.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """The five memory tiers."""

    WORKING = "working"  # run-local scratch variables
    SESSION = "session"  # sliding conversation window
    SEMANTIC = "semantic"  # durable facts
    EPISODIC = "episodic"  # events, especially failures
    PROCEDURAL = "procedural"  # SOPs / how-to


class MemoryStatus(StrEnum):
    """Lifecycle states. Transitions are governed in Phase 2.

    ``CANDIDATE -> VALIDATING -> ACTIVE -> SUPERSEDED -> ARCHIVED -> DELETED``
    """

    CANDIDATE = "candidate"
    VALIDATING = "validating"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class TrustLevel(StrEnum):
    """Provenance trust label. Everything at/below ``MEMORY`` is untrusted data."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    MEMORY = "memory"
    TOOL_RESULT = "tool_result"
    RAG_CONTENT = "rag_content"


def _new_id() -> str:
    return f"mem_{uuid.uuid4().hex[:24]}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryRecord(BaseModel):
    """A single memory with full metadata and provenance edges."""

    id: str = Field(default_factory=_new_id)
    type: MemoryType
    content: str = Field(min_length=1)
    status: MemoryStatus = MemoryStatus.CANDIDATE

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="user", description="e.g. 'user', 'tool:metrics_api'")
    trust: TrustLevel = TrustLevel.USER

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    version: int = Field(default=1, ge=1)

    # Provenance graph edges (Phase 2 exposes these via /memory/{id}/provenance).
    supersedes: str | None = None
    derived_from: list[str] = Field(default_factory=list)

    def touch(self) -> None:
        """Advance ``updated_at`` to now (called on any mutation)."""
        self.updated_at = _utcnow()


class MemoryCreate(BaseModel):
    """Input schema for creating a memory. Server assigns id/timestamps/version."""

    type: MemoryType
    content: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "user"
    trust: TrustLevel = TrustLevel.USER
    status: MemoryStatus = MemoryStatus.CANDIDATE
    supersedes: str | None = None
    derived_from: list[str] = Field(default_factory=list)


class MemoryUpdate(BaseModel):
    """Partial update. Only provided fields are changed; version is bumped."""

    content: str | None = Field(default=None, min_length=1)
    status: MemoryStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    trust: TrustLevel | None = None
    supersedes: str | None = None
    derived_from: list[str] | None = None
