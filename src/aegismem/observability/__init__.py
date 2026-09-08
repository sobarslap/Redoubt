"""AegisMem observability subsystem — reproducibility, not just logging (Phase 7).

An always-on local JSONL trace store is the source of truth; Langfuse (optional)
is a derived exporter. Every run captures correlation IDs, per-stage spans with
their decision attributes, and recorded LLM/tool calls — enough to *replay* why a
run behaved as it did, via ``aegismem replay <run_id>``.
"""

from aegismem.observability.events import (
    CallKind,
    CallRecord,
    RunRecord,
    SpanRecord,
    SpanStatus,
)
from aegismem.observability.langfuse_exporter import LangfuseExporter
from aegismem.observability.replay import (
    ReplayCache,
    ReplayedRun,
    Replayer,
    TimelineStep,
)
from aegismem.observability.trace_store import (
    FanoutSink,
    JSONLTraceStore,
    RunTrace,
    TraceSink,
)
from aegismem.observability.tracer import RunTrace as RunTraceContext
from aegismem.observability.tracer import SpanHandle, Tracer

__all__ = [
    "CallKind",
    "CallRecord",
    "FanoutSink",
    "JSONLTraceStore",
    "LangfuseExporter",
    "ReplayCache",
    "ReplayedRun",
    "Replayer",
    "RunRecord",
    "RunTrace",
    "RunTraceContext",
    "SpanHandle",
    "SpanRecord",
    "SpanStatus",
    "TimelineStep",
    "TraceSink",
    "Tracer",
]
