"""Runtime-contract request/response schemas.

Every request enters and every response leaves through these typed envelopes.
Correlation IDs thread client -> engine -> observability:
``request_id`` (client) -> ``run_id`` (engine) -> ``trace_id`` (tracing).

The execution stages that populate ``citations``, ``usage``, and ``timings``
land in later phases; the contract is fixed now so those phases fill it in
rather than reshaping the boundary.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field

from aegismem.errors import ErrorEnvelope


class RunMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


class AgentRequest(BaseModel):
    request_id: str = Field(default_factory=_request_id)
    session_id: str
    input: str = Field(min_length=1)
    mode: RunMode = RunMode.SYNC
    idempotency_key: str | None = None
    budget_overrides: dict[str, float] | None = None


class Citation(BaseModel):
    """A grounding pointer from an answer back to the memory it used."""

    memory_id: str
    score: float | None = None
    snippet: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0


class Timings(BaseModel):
    """Per-stage wall-clock milliseconds; keys mirror the runtime-contract stages."""

    total_ms: float = 0.0
    stages: dict[str, float] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    request_id: str
    run_id: str
    status: RunStatus
    output: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    timings: Timings = Field(default_factory=Timings)
    trace_id: str | None = None
    error: ErrorEnvelope | None = None
