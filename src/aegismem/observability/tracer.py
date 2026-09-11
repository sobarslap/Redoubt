"""Tracer — threads correlation IDs through the pipeline and records everything.

Usage mirrors the runtime contract's stage boundaries::

    tracer = Tracer(sink)
    with tracer.run(request) as run:
        with run.span("guardrails") as sp:
            sp.set(injection_score=v.score, families=v.families)
        answer = run.record_llm("plan", {"prompt": p}, produce=lambda: llm(p))
        run.set_output(text, usage=Usage(...))

Two guarantees:

* **Fail-open** — per ``failure_policy.observability: continue``, no tracing call
  ever raises into the run. A broken sink increments a drop counter; the run
  proceeds.
* **Replayable** — ``record_llm`` / ``record_tool`` capture the request key and
  response. Attach a :class:`ReplayCache` (from a prior run's trace) and the same
  calls are served from the recording instead of re-executed, making
  non-deterministic behavior deterministic on replay.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING

from aegismem.api.models import Usage
from aegismem.errors import AegisError, ErrorEnvelope
from aegismem.observability.events import (
    CallKind,
    CallRecord,
    RunRecord,
    SpanRecord,
    SpanStatus,
    _trace_id,
)

if TYPE_CHECKING:
    from aegismem.api.models import AgentRequest
    from aegismem.observability.replay import ReplayCache
    from aegismem.observability.trace_store import TraceSink


def _run_id() -> str:
    return f"run_{uuid.uuid4().hex[:24]}"


class SpanHandle:
    """A stage span. ``set`` attaches attributes; exit records duration/status."""

    def __init__(self, run: RunTrace, name: str, seq: int) -> None:
        self._run = run
        self._name = name
        self._seq = seq
        self._attrs: dict[str, object] = {}
        self._start = time.perf_counter()
        self._status = SpanStatus.OK
        self._error: ErrorEnvelope | None = None

    def set(self, **attrs: object) -> None:
        self._attrs.update(attrs)

    def __enter__(self) -> SpanHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        duration = (time.perf_counter() - self._start) * 1000
        if exc is not None:
            self._status = SpanStatus.ERROR
            if isinstance(exc, AegisError):
                self._error = exc.envelope
        self._run._emit(
            SpanRecord(
                run_id=self._run.run_id,
                trace_id=self._run.trace_id,
                name=self._name,
                seq=self._seq,
                duration_ms=round(duration, 4),
                status=self._status,
                attributes=self._attrs,
                error=self._error,
            )
        )
        # Do not suppress exceptions — the run's own error handling owns them.


class RunTrace:
    def __init__(
        self,
        sink: TraceSink,
        *,
        run_id: str,
        request_id: str,
        trace_id: str,
        session_id: str,
        input_text: str,
        cache: ReplayCache | None = None,
    ) -> None:
        self._sink = sink
        self.run_id = run_id
        self.trace_id = trace_id
        self._record = RunRecord(
            run_id=run_id,
            request_id=request_id,
            trace_id=trace_id,
            session_id=session_id,
            input=input_text,
        )
        self._cache = cache
        self._seq = 0
        self._start = time.perf_counter()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, record: object) -> None:
        with contextlib.suppress(Exception):  # fail-open per failure policy
            self._sink.emit(record)  # type: ignore[arg-type]

    # -- spans ---------------------------------------------------------------

    def span(self, name: str) -> SpanHandle:
        return SpanHandle(self, name, self._next_seq())

    # -- recorded (replayable) calls ----------------------------------------

    def record_llm(self, key: str, request: dict[str, object], produce: Callable[[], str]) -> str:
        return self._record_call(CallKind.LLM, key, request, produce)

    def record_tool(self, key: str, request: dict[str, object], produce: Callable[[], str]) -> str:
        return self._record_call(CallKind.TOOL, key, request, produce)

    def _record_call(
        self,
        call_kind: CallKind,
        key: str,
        request: dict[str, object],
        produce: Callable[[], str],
    ) -> str:
        seq = self._next_seq()
        if self._cache is not None and self._cache.has(call_kind, key):
            response = self._cache.get(call_kind, key)
            latency = 0.0
            failed = False
        else:
            start = time.perf_counter()
            failed = False
            try:
                response = produce()
            except Exception:
                failed = True
                response = ""
                latency = (time.perf_counter() - start) * 1000
                self._emit(
                    CallRecord(
                        run_id=self.run_id,
                        trace_id=self.trace_id,
                        call_kind=call_kind,
                        seq=seq,
                        key=key,
                        request=request,
                        response=response,
                        latency_ms=round(latency, 4),
                        failed=True,
                    )
                )
                raise
            latency = (time.perf_counter() - start) * 1000
        self._emit(
            CallRecord(
                run_id=self.run_id,
                trace_id=self.trace_id,
                call_kind=call_kind,
                seq=seq,
                key=key,
                request=request,
                response=response,
                latency_ms=round(latency, 4),
                failed=failed,
            )
        )
        return response

    # -- outcome -------------------------------------------------------------

    def set_output(self, outcome: str, *, usage: Usage | None = None) -> None:
        self._record.outcome = outcome[:500]
        if usage is not None:
            self._record.usage = usage

    def fail(self, error: AegisError) -> None:
        self._record.status = "failed"
        self._record.error = error.envelope

    def __enter__(self) -> RunTrace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._record.total_ms = round((time.perf_counter() - self._start) * 1000, 4)
        from aegismem.observability.events import _utcnow_iso

        self._record.ended_at = _utcnow_iso()
        if exc is not None and self._record.status != "failed":
            self._record.status = "failed"
            if isinstance(exc, AegisError):
                self._record.error = exc.envelope
        elif self._record.status == "running":
            self._record.status = "succeeded"
        self._emit(self._record)


class Tracer:
    """Opens :class:`RunTrace` contexts. Sink is a local store, a fanout, or an
    exporter — anything with ``emit``."""

    def __init__(self, sink: TraceSink) -> None:
        self._sink = sink

    def run(self, request: AgentRequest, *, cache: ReplayCache | None = None) -> RunTrace:
        return RunTrace(
            self._sink,
            run_id=_run_id(),
            request_id=request.request_id,
            trace_id=_trace_id(),
            session_id=request.session_id,
            input_text=request.input,
            cache=cache,
        )
