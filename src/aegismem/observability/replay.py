"""Deterministic replay — reconstruct *why* a run behaved as it did.

Given a ``run_id``, the trace store hands back the run's records; this module
turns them into:

* a **timeline** — the ordered pipeline the run executed (guardrails ->
  retrieval -> ... -> execution -> output), with each span's key attributes, so a
  human or the CLI can see the decisions, and
* a **ReplayCache** — the recorded LLM/tool responses keyed by request, so
  re-running the same pipeline with this cache attached produces identical
  non-deterministic calls. That is what makes an agent run debuggable rather than
  a one-off.
"""

from __future__ import annotations

from pydantic import BaseModel

from aegismem.observability.events import CallKind, SpanRecord
from aegismem.observability.trace_store import JSONLTraceStore, RunTrace


def _cache_key(kind: CallKind, key: str) -> str:
    return f"{kind.value}:{key}"


class ReplayCache:
    """Serves recorded call responses by ``(kind, key)``. Built from a run's
    ``CallRecord``s; attach to ``Tracer.run(cache=...)`` for deterministic replay."""

    def __init__(self, responses: dict[str, str], failures: dict[str, bool] | None = None) -> None:
        self._responses = responses
        self._failures = failures or {}

    @classmethod
    def from_trace(cls, trace: RunTrace) -> ReplayCache:
        return cls(
            {_cache_key(c.call_kind, c.key): c.response for c in trace.calls},
            {_cache_key(c.call_kind, c.key): c.failed for c in trace.calls},
        )

    def has(self, kind: CallKind, key: str) -> bool:
        return _cache_key(kind, key) in self._responses

    def get(self, kind: CallKind, key: str) -> str:
        return self._responses[_cache_key(kind, key)]

    def failed(self, kind: CallKind, key: str) -> bool:
        """Whether the recorded call originally failed (so replay reproduces it)."""
        return self._failures.get(_cache_key(kind, key), False)


class TimelineStep(BaseModel):
    seq: int
    name: str
    duration_ms: float
    status: str
    attributes: dict[str, object] = {}


class ReplayedRun(BaseModel):
    run_id: str
    request_id: str
    trace_id: str
    status: str
    total_ms: float
    input: str
    outcome: str | None
    timeline: list[TimelineStep]
    calls: int


class Replayer:
    def __init__(self, store: JSONLTraceStore) -> None:
        self._store = store

    def load(self, run_id: str) -> RunTrace:
        return self._store.load(run_id)

    def cache(self, run_id: str) -> ReplayCache:
        return ReplayCache.from_trace(self._store.load(run_id))

    def reconstruct(self, run_id: str) -> ReplayedRun:
        trace = self._store.load(run_id)
        return ReplayedRun(
            run_id=trace.run.run_id,
            request_id=trace.run.request_id,
            trace_id=trace.run.trace_id,
            status=trace.run.status,
            total_ms=trace.run.total_ms,
            input=trace.run.input,
            outcome=trace.run.outcome,
            timeline=[_step(s) for s in trace.spans],
            calls=len(trace.calls),
        )


def _step(span: SpanRecord) -> TimelineStep:
    return TimelineStep(
        seq=span.seq,
        name=span.name,
        duration_ms=span.duration_ms,
        status=span.status.value,
        attributes=span.attributes,
    )
