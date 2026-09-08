"""Optional Langfuse exporter — a *derived* view of the local trace store.

Langfuse gives the span-waterfall UI for the DevOps demo, but it is never the
source of truth: the local JSONL store is (it powers replay and survives with no
network). This exporter is import-guarded (Langfuse lives in the ``observability``
dependency group) and fully fire-and-forget — if the package is absent or the
client errors, ``emit`` is a silent no-op, honoring ``failure_policy:
observability: continue``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aegismem.observability.events import CallRecord, RunRecord, SpanRecord


class LangfuseExporter:
    """Mirrors trace records into Langfuse when available; no-op otherwise."""

    def __init__(self, client: Any | None = None) -> None:
        self._client: Any = client if client is not None else self._try_build_client()
        self.available: bool = self._client is not None

    @staticmethod
    def _try_build_client() -> Any | None:
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except Exception:
            return None
        try:
            return Langfuse()
        except Exception:  # missing keys / config — degrade to no-op
            return None

    def emit(self, record: BaseModel) -> None:
        if not self.available:
            return
        try:
            self._export(record)
        except Exception:  # never let the exporter affect the run
            return

    def _export(self, record: BaseModel) -> None:
        client = self._client
        if isinstance(record, RunRecord):
            client.trace(
                id=record.trace_id,
                name="aegismem.run",
                input=record.input,
                output=record.outcome,
                metadata={"run_id": record.run_id, "status": record.status},
            )
        elif isinstance(record, SpanRecord):
            client.span(
                trace_id=record.trace_id,
                name=record.name,
                metadata=record.attributes,
            )
        elif isinstance(record, CallRecord):
            client.generation(
                trace_id=record.trace_id,
                name=record.call_kind.value,
                input=record.request,
                output=record.response,
            )
