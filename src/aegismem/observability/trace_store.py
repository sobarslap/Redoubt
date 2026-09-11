"""Local trace store — the authoritative, always-on record that powers replay.

An append-only JSONL log is the source of truth for observability (Langfuse, when
configured, is a *derived* exporter — see ``langfuse_exporter``). One line per
record, discriminated by ``kind``. Writes are best-effort per the failure policy
(``observability: continue``): a write that fails is buffered/dropped, never
raised, so tracing can never take down a run.

Reads filter the log by ``run_id`` and return the run's records for replay. At
dev/demo scale (the documented single-node tier) a flat JSONL scan is plenty; the
same interface fronts a SQLite-backed store on the production swap path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from aegismem.observability.events import (
    CallRecord,
    RecordKind,
    RunRecord,
    SpanRecord,
)

_Record = RunRecord | SpanRecord | CallRecord


@runtime_checkable
class TraceSink(Protocol):
    """Anything a run can emit records to (local store, Langfuse exporter)."""

    def emit(self, record: BaseModel) -> None: ...


class RunTrace(BaseModel):
    """A run's records, reassembled for inspection / replay."""

    run: RunRecord
    spans: list[SpanRecord]
    calls: list[CallRecord]


class JSONLTraceStore:
    """Append-only JSONL trace store. ``emit`` never raises."""

    def __init__(self, path: str | Path = "traces/trace.jsonl") -> None:
        self._path = Path(path)
        self.dropped = 0  # records lost to write failures (fire-and-forget)

    def emit(self, record: BaseModel) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = record.model_dump_json()
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:  # failure policy: continue, do not raise
            self.dropped += 1

    # -- read side (replay) --------------------------------------------------

    def _iter_lines(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        rows: list[dict[str, object]] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A truncated tail record (e.g. a crash mid-write) must not make
                    # the whole trace unreadable — skip the malformed line.
                    self.dropped += 1
        return rows

    def load(self, run_id: str) -> RunTrace:
        run: RunRecord | None = None
        spans: list[SpanRecord] = []
        calls: list[CallRecord] = []
        for row in self._iter_lines():
            if row.get("run_id") != run_id:
                continue
            kind = row.get("kind")
            if kind == RecordKind.RUN:
                run = RunRecord.model_validate(row)  # last one wins (final state)
            elif kind == RecordKind.SPAN:
                spans.append(SpanRecord.model_validate(row))
            elif kind == RecordKind.CALL:
                calls.append(CallRecord.model_validate(row))
        if run is None:
            from aegismem.errors import NotFoundError

            raise NotFoundError(f"run {run_id!r} not found in trace store", stage="observability")
        spans.sort(key=lambda s: s.seq)
        calls.sort(key=lambda c: c.seq)
        return RunTrace(run=run, spans=spans, calls=calls)

    def run_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self._iter_lines():
            if row.get("kind") == RecordKind.RUN:
                rid = str(row.get("run_id"))
                seen[rid] = None
        return list(seen)


class FanoutSink:
    """Emit to several sinks; one sink's failure never affects the others."""

    def __init__(self, sinks: list[TraceSink]) -> None:
        self._sinks = sinks

    def emit(self, record: BaseModel) -> None:
        for sink in self._sinks:
            try:
                sink.emit(record)
            except Exception:  # isolate a misbehaving exporter
                continue
