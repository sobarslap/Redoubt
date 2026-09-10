"""Production Phase P7 gate — observability & ops.

Proves the service exposes Prometheus metrics that move with real runs, that the
exported trace copy is PII-redacted before leaving the process, and that a run
executed through the service is replayable by run_id from the local store.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from aegismem.execution.agent import AgentRuntime
from aegismem.execution.llm import MockProvider
from aegismem.guardrails import SecurityBoundary
from aegismem.observability.events import RunRecord, SpanRecord
from aegismem.observability.metrics import MetricsRegistry
from aegismem.observability.redact import RedactingSink
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex


def _app(tmp_path, metrics):
    from aegismem.api.app import create_app

    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    rt = AgentRuntime(
        llm=MockProvider(scripts=[("checkout", "restart and scale replicas")]),
        router=router,
        tracer=Tracer(JSONLTraceStore(tmp_path / "svc.jsonl")),
        boundary=SecurityBoundary(),
    )
    rt.remember("mem_sop", "runbook for the checkout service")
    return create_app(rt, metrics=metrics)


# -- metrics -----------------------------------------------------------------


def test_metrics_endpoint_moves_with_runs(tmp_path) -> None:
    metrics = MetricsRegistry()
    c = TestClient(_app(tmp_path, metrics))
    before = c.get("/metrics").text
    assert "aegismem_runs_total" in before
    c.post("/runs", json={"session_id": "s", "input": "why is checkout erroring?"})
    after = c.get("/metrics").text
    assert "aegismem_runs_total 1.0" in after
    assert 'aegismem_run_latency_ms{quantile="0.95"}' in after


def test_metrics_render_is_prometheus_text() -> None:
    m = MetricsRegistry()
    m.inc("aegismem_runs_total", 3)
    m.observe("aegismem_run_latency_ms", 12.0)
    m.observe("aegismem_run_latency_ms", 20.0)
    text = m.render()
    assert "# TYPE aegismem_runs_total counter" in text
    assert "aegismem_run_latency_ms_count 2" in text
    assert "aegismem_run_latency_ms_sum 32.0" in text


# -- PII-safe export redaction -----------------------------------------------


def test_redacting_sink_scrubs_secrets_before_export() -> None:
    captured: list[object] = []

    class _Capture:
        def emit(self, record) -> None:
            captured.append(record)

    sink = RedactingSink(_Capture())
    sink.emit(
        RunRecord(
            run_id="r",
            request_id="q",
            trace_id="t",
            input="key sk-ABCD1234ABCD1234ABCD leaked",
            outcome="ops@example.com noted",
        )
    )
    sink.emit(
        SpanRecord(
            run_id="r",
            trace_id="t",
            name="execution",
            attributes={"note": "token AKIA1234567890ABCD00", "n": 5},
        )
    )
    run = captured[0]
    span = captured[1]
    assert "sk-ABCD1234ABCD1234ABCD" not in run.input
    assert "ops@example.com" not in run.outcome
    assert "AKIA1234567890ABCD00" not in span.attributes["note"]
    assert span.attributes["n"] == 5  # non-string attrs untouched


# -- replay of a service run -------------------------------------------------


def test_service_run_is_replayable_by_run_id(tmp_path) -> None:
    from aegismem.observability.replay import Replayer

    store = JSONLTraceStore(tmp_path / "svc.jsonl")
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    rt = AgentRuntime(
        llm=MockProvider(scripts=[("checkout", "ok")]),
        router=router,
        tracer=Tracer(store),
        boundary=SecurityBoundary(),
    )
    rt.remember("mem_sop", "runbook for checkout")
    from aegismem.api.app import create_app

    c = TestClient(create_app(rt))
    run_id = c.post("/runs", json={"session_id": "s", "input": "checkout?"}).json()["run_id"]
    recon = Replayer(store).reconstruct(run_id)
    assert [s.name for s in recon.timeline] == ["guardrails", "retrieval", "execution", "output"]
