"""Phase 7 gate — observability = reproducibility.

Asserts the trace captures correlation IDs, per-stage spans with their decision
attributes, latency/tokens/outcome; that tracing is fail-open (a broken sink
never breaks the run); and that a run replays deterministically — recorded
LLM/tool responses are served back from the trace instead of re-executed.
"""

from __future__ import annotations

from pydantic import BaseModel

from aegismem.api.models import AgentRequest, Usage
from aegismem.errors import ToolExecutionError
from aegismem.observability.events import CallKind
from aegismem.observability.replay import ReplayCache, Replayer
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer


def _request(text: str = "checkout is erroring") -> AgentRequest:
    return AgentRequest(session_id="sess_1", input=text)


def _tracer(tmp_path) -> tuple[Tracer, JSONLTraceStore]:
    store = JSONLTraceStore(tmp_path / "trace.jsonl")
    return Tracer(store), store


# -- capture -----------------------------------------------------------------


def test_run_captures_ids_spans_and_usage(tmp_path) -> None:
    tracer, store = _tracer(tmp_path)
    req = _request()
    with tracer.run(req) as run:
        run_id = run.run_id
        with run.span("guardrails") as sp:
            sp.set(injection_score=0.0, blocked=False)
        with run.span("retrieval") as sp:
            sp.set(strategy="hybrid", retrieved=["mem_a", "mem_b"])
        run.set_output("proposed rollback", usage=Usage(input_tokens=120, output_tokens=40))

    trace = store.load(run_id)
    assert trace.run.request_id == req.request_id
    assert trace.run.trace_id.startswith("trace_")
    assert trace.run.status == "succeeded"
    assert trace.run.usage.input_tokens == 120
    assert [s.name for s in trace.spans] == ["guardrails", "retrieval"]
    assert trace.spans[1].attributes["strategy"] == "hybrid"
    assert trace.run.total_ms >= 0.0


def test_span_records_error_status_on_exception(tmp_path) -> None:
    tracer, store = _tracer(tmp_path)
    with tracer.run(_request()) as run:
        run_id = run.run_id
        try:
            with run.span("execution"):
                raise ToolExecutionError("tool blew up")
        except ToolExecutionError:
            pass
    trace = store.load(run_id)
    span = trace.spans[0]
    assert span.status.value == "error"
    assert span.error is not None and span.error.category.value == "tool"


# -- fail-open ---------------------------------------------------------------


class _BrokenSink(BaseModel):
    def emit(self, record: object) -> None:
        raise RuntimeError("sink is down")


def test_tracing_is_fail_open(tmp_path) -> None:
    tracer = Tracer(_BrokenSink())
    # A broken sink must not break the run.
    with tracer.run(_request()) as run:
        with run.span("guardrails") as sp:
            sp.set(ok=True)
        out = run.record_llm("k", {"p": "x"}, produce=lambda: "answer")
    assert out == "answer"


# -- deterministic replay ----------------------------------------------------


def test_recorded_call_replays_without_reexecuting(tmp_path) -> None:
    tracer, store = _tracer(tmp_path)
    calls = {"n": 0}

    def produce() -> str:
        calls["n"] += 1
        return f"llm-output-{calls['n']}"

    with tracer.run(_request()) as run:
        run_id = run.run_id
        first = run.record_llm("plan", {"prompt": "diagnose"}, produce=produce)
    assert first == "llm-output-1"
    assert calls["n"] == 1

    # Replay: same pipeline, cache attached -> the producer must NOT run again.
    cache = Replayer(store).cache(run_id)
    with tracer.run(_request(), cache=cache) as run2:
        replayed = run2.record_llm("plan", {"prompt": "diagnose"}, produce=produce)
    assert replayed == "llm-output-1"  # identical to the recorded response
    assert calls["n"] == 1  # producer was not invoked on replay


def test_reconstruct_returns_ordered_timeline(tmp_path) -> None:
    tracer, store = _tracer(tmp_path)
    with tracer.run(_request()) as run:
        run_id = run.run_id
        with run.span("guardrails"):
            pass
        run.record_tool("read_logs", {"svc": "checkout"}, produce=lambda: "log lines")
        with run.span("execution"):
            pass
        run.set_output("done")

    recon = Replayer(store).reconstruct(run_id)
    assert [s.name for s in recon.timeline] == ["guardrails", "execution"]
    assert recon.calls == 1
    assert recon.status == "succeeded"


def test_failed_call_is_recorded_and_reraised(tmp_path) -> None:
    tracer, store = _tracer(tmp_path)

    def boom() -> str:
        raise ToolExecutionError("timeout")

    import pytest

    with tracer.run(_request()) as run:
        run_id = run.run_id
        with pytest.raises(ToolExecutionError):
            run.record_tool("restart", {"svc": "db"}, produce=boom)

    trace = store.load(run_id)
    assert len(trace.calls) == 1
    assert trace.calls[0].failed is True
    assert trace.calls[0].call_kind == CallKind.TOOL


# -- replay cache primitive --------------------------------------------------


def test_replay_cache_serves_by_kind_and_key() -> None:
    cache = ReplayCache({"llm:plan": "cached"})
    assert cache.has(CallKind.LLM, "plan")
    assert cache.get(CallKind.LLM, "plan") == "cached"
    assert not cache.has(CallKind.TOOL, "plan")
