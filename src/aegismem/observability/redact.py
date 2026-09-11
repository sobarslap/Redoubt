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
    def _redact(record: BaseModel) -> BaseModel:
        if isinstance(record, RunRecord):
            return record.model_copy(
                update={
                    "input": pii.redact(record.input),
                    "outcome": pii.redact(record.outcome) if record.outcome else record.outcome,
                }
            )
        if isinstance(record, SpanRecord):
            attrs = {
                k: (pii.redact(v) if isinstance(v, str) else v)
                for k, v in record.attributes.items()
            }
            return record.model_copy(update={"attributes": attrs})
        if isinstance(record, CallRecord):
            return record.model_copy(update={"response": pii.redact(record.response)})
        return record
