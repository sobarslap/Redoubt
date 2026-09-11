"""Typed trace records — the schema of what every run captures.

Observability here is *reproducibility*, not just logging: the records carry
enough to reconstruct **why** a run behaved as it did, not merely that it did.
Three record kinds share one append-only log, discriminated by ``kind``:

* ``RunRecord``  — one per run: correlation IDs, input, usage, outcome, timing.
* ``SpanRecord`` — one per pipeline stage: name, duration, attributes, status.
* ``CallRecord`` — one per non-deterministic call (LLM / tool): the request key
  and the recorded response, so replay can serve it back deterministically.

Correlation threads client -> engine -> tracing:
``request_id`` -> ``run_id`` -> ``trace_id`` (mirrors ``api.models``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from aegismem.api.models import Usage
from aegismem.errors import ErrorEnvelope


class RecordKind(StrEnum):
    RUN = "run"
    SPAN = "span"
    CALL = "call"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class CallKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:24]}"


def _span_id() -> str:
    return f"span_{uuid.uuid4().hex[:16]}"


class RunRecord(BaseModel):
    kind: RecordKind = RecordKind.RUN
    run_id: str
    request_id: str
    trace_id: str
    session_id: str = ""
    input: str = ""
    status: str = "running"
    outcome: str | None = None  # short summary of the final response
    usage: Usage = Field(default_factory=Usage)
    total_ms: float = 0.0
    started_at: str = Field(default_factory=_utcnow_iso)
    ended_at: str | None = None
    error: ErrorEnvelope | None = None


class SpanRecord(BaseModel):
    kind: RecordKind = RecordKind.SPAN
    span_id: str = Field(default_factory=_span_id)
    run_id: str
    trace_id: str
    name: str  # pipeline stage, e.g. "guardrails", "retrieval", "execution"
    seq: int = 0  # monotonic order within the run
    duration_ms: float = 0.0
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, object] = Field(default_factory=dict)
    error: ErrorEnvelope | None = None
    started_at: str = Field(default_factory=_utcnow_iso)


class CallRecord(BaseModel):
    kind: RecordKind = RecordKind.CALL
    run_id: str
    trace_id: str
    call_kind: CallKind
    seq: int = 0
    key: str  # deterministic key: identical inputs -> identical key
    request: dict[str, object] = Field(default_factory=dict)
    response: str = ""
    latency_ms: float = 0.0
    failed: bool = False
