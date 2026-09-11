"""PII-safe trace redaction (Production P7).

Traces are exported to a hosted store (Langfuse); before they leave the process,
this sink scrubs secrets and PII from the fields that carry free text — a run's
input/outcome and any span attribute values — so credentials logged in a poisoned
tool result never land in the observability backend. Wrap any sink:

    FanoutSink([JSONLTraceStore(...), RedactingSink(LangfuseExporter())])

The local JSONL store can stay un-redacted (it powers exact replay and never
leaves the host); only the exported copy is scrubbed.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from aegismem.guardrails import pii
from aegismem.observability.events import CallRecord, RunRecord, SpanRecord
from aegismem.observability.trace_store import TraceSink


@dataclass
class RedactingSink:
    inner: TraceSink

    def emit(self, record: BaseModel) -> None:
        self.inner.emit(self._redact(record))

    @staticmethod
    def _redact_value(value: object) -> object:
        """Recursively scrub strings inside str / list / dict values."""
        if isinstance(value, str):
            return pii.redact(value)
        if isinstance(value, list):
            return [RedactingSink._redact_value(v) for v in value]
        if isinstance(value, dict):
            return {k: RedactingSink._redact_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _redact(record: BaseModel) -> BaseModel:
        if isinstance(record, RunRecord):
            return record.model_copy(
                update={
                    "input": pii.redact(record.input),
                    "outcome": pii.redact(record.outcome) if record.outcome else record.outcome,
                }
            )
        if isinstance(record, SpanRecord):
            # Scrub every attribute value, including lists/dicts (not just top-level str).
            attrs = {k: RedactingSink._redact_value(v) for k, v in record.attributes.items()}
            return record.model_copy(update={"attributes": attrs})
        if isinstance(record, CallRecord):
            # Tool-call args (request) can carry secrets too — scrub both sides.
            request = {k: RedactingSink._redact_value(v) for k, v in record.request.items()}
            return record.model_copy(
                update={"request": request, "response": pii.redact(record.response)}
            )
        return record
