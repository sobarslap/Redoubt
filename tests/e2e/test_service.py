"""Production Phase P3 gate — the FastAPI service honors the runtime contract.

Covers sync + async runs, idempotency, cancellation, backpressure, the typed
error envelope on every path, and a published OpenAPI schema. Runs fully in-proc
(no external service) via FastAPI's TestClient.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from aegismem.api.app import create_app
from aegismem.api.models import AgentRequest, RunStatus
from aegismem.execution.agent import AgentRuntime
from aegismem.execution.llm import MockProvider
from aegismem.guardrails import SecurityBoundary
from aegismem.memory.store import SQLiteMemoryStore
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex


def _runtime(tmp_path) -> AgentRuntime:
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    llm = MockProvider(scripts=[("checkout", "Restart the service and scale replicas.")])
    rt = AgentRuntime(
        llm=llm,
        router=router,
        tracer=Tracer(JSONLTraceStore(tmp_path / "t.jsonl")),
        boundary=SecurityBoundary(),
    )
    rt.remember("mem_sop", "Runbook: restart the service on pool exhaustion for checkout")
    return rt


def _client(tmp_path, **kw) -> TestClient:
    store = kw.pop("provenance_store", None)
    return TestClient(
        create_app(_runtime(tmp_path), trace_store=None, provenance_store=store, **kw)
    )


# -- health + schema ---------------------------------------------------------


def test_health_and_ready(tmp_path) -> None:
    c = _client(tmp_path)
    assert c.get("/healthz").json()["status"] == "ok"
    assert c.get("/readyz").json()["status"] == "ready"


def test_openapi_is_published_with_runs(tmp_path) -> None:
    schema = _client(tmp_path).get("/openapi.json").json()
    assert "/runs" in schema["paths"]
    assert "/runs/{run_id}" in schema["paths"]


# -- sync run ----------------------------------------------------------------


def test_sync_run_returns_grounded_response(tmp_path) -> None:
    c = _client(tmp_path)
    r = c.post("/runs", json={"session_id": "s", "input": "why is checkout erroring?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == RunStatus.SUCCEEDED
    assert body["output"]
    assert body["trace_id"]
    # And it is retrievable by run_id.
    got = c.get(f"/runs/{body['run_id']}")
    assert got.status_code == 200
    assert got.json()["run_id"] == body["run_id"]


def test_idempotency_key_returns_same_run(tmp_path) -> None:
    c = _client(tmp_path)
    payload = {"session_id": "s", "input": "checkout status?", "idempotency_key": "k-1"}
    first = c.post("/runs", json=payload).json()
    second = c.post("/runs", json=payload).json()
    assert first["run_id"] == second["run_id"]


# -- async run ---------------------------------------------------------------


def test_async_run_queues_then_completes(tmp_path) -> None:
    # Context-manager form keeps the event loop alive so the background worker runs.
    with TestClient(create_app(_runtime(tmp_path))) as c:
        r = c.post("/runs", json={"session_id": "s", "input": "checkout?", "mode": "async"})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        assert r.json()["status"] == RunStatus.QUEUED

        for _ in range(100):  # poll until the background worker finishes
            body = c.get(f"/runs/{run_id}").json()
            if body["status"] != RunStatus.QUEUED:
                break
            time.sleep(0.02)
    assert body["status"] == RunStatus.SUCCEEDED
    assert body["output"]


# -- cancellation ------------------------------------------------------------


def test_cancel_endpoint_marks_run_cancelled(tmp_path) -> None:
    c = _client(tmp_path)
    r = c.post("/runs", json={"session_id": "s", "input": "checkout?", "mode": "async"})
    run_id = r.json()["run_id"]
    assert c.delete(f"/runs/{run_id}").json()["cancelled"] is True
    # The accepted DELETE is not enough on its own — assert the safety property it
    # buys: a cancelled run must never resolve to a successful completion with a
    # grounded answer. Poll the run and confirm it never reports SUCCEEDED/output.
    for _ in range(50):
        body = c.get(f"/runs/{run_id}").json()
        assert body["status"] != RunStatus.SUCCEEDED
        assert not body.get("output")
        if body["status"] == RunStatus.FAILED:
            break


def test_runtime_cancellation_at_stage_boundary(tmp_path) -> None:
    # Cooperative cancellation is enforced in the runtime itself.
    rt = _runtime(tmp_path)
    resp = rt.handle(AgentRequest(session_id="s", input="checkout?"), cancel_check=lambda: True)
    assert resp.status is RunStatus.FAILED
    assert resp.error is not None and resp.error.code == "cancelled"


# -- backpressure + error envelope -------------------------------------------


def test_backpressure_returns_typed_503(tmp_path) -> None:
    c = _client(tmp_path, max_in_flight=0)  # every request is over capacity
    r = c.post("/runs", json={"session_id": "s", "input": "checkout?"})
    assert r.status_code == 503
    env = r.json()
    assert env["category"] == "budget" and env["retryable"] is True


def test_caller_cannot_access_another_callers_run(tmp_path) -> None:
    # A run belongs to the principal that created it; another authenticated caller
    # can neither read nor cancel it (nor confirm it exists) -> 404.
    from aegismem.api.app import create_app
    from aegismem.security.auth import ApiKeyStore, Principal

    keys = ApiKeyStore.from_mapping(
        {"key-a": Principal("alice", "tenant-a"), "key-b": Principal("bob", "tenant-b")}
    )
    c = TestClient(create_app(_runtime(tmp_path), api_keys=keys))
    made = c.post(
        "/runs", json={"session_id": "s", "input": "checkout?"}, headers={"X-API-Key": "key-a"}
    ).json()
    run_id = made["run_id"]
    # Owner can read it.
    assert c.get(f"/runs/{run_id}", headers={"X-API-Key": "key-a"}).status_code == 200
    # A different caller cannot read or cancel it.
    assert c.get(f"/runs/{run_id}", headers={"X-API-Key": "key-b"}).status_code == 404
    assert c.delete(f"/runs/{run_id}", headers={"X-API-Key": "key-b"}).status_code == 404


def test_unknown_run_returns_typed_404(tmp_path) -> None:
    r = _client(tmp_path).get("/runs/run_does_not_exist")
    assert r.status_code == 404
    env = r.json()
    assert env["code"] == "not_found" and env["category"] == "not_found"


def test_validation_error_is_422(tmp_path) -> None:
    # Missing required 'input' -> FastAPI validation (typed 422).
    r = _client(tmp_path).post("/runs", json={"session_id": "s"})
    assert r.status_code == 422


# -- provenance passthrough --------------------------------------------------


def test_provenance_endpoint(tmp_path) -> None:
    from aegismem.memory.models import MemoryCreate, MemoryType, TrustLevel

    store = SQLiteMemoryStore(":memory:")
    a = store.create(MemoryCreate(type=MemoryType.SEMANTIC, content="old", trust=TrustLevel.USER))
    b = store.create(MemoryCreate(type=MemoryType.SEMANTIC, content="new", trust=TrustLevel.USER))
    store.add_edge(b.id, a.id, "supersedes")
    c = _client(tmp_path, provenance_store=store)
    r = c.get(f"/memory/{b.id}/provenance")
    assert r.status_code == 200
    assert a.id in r.json()["supersedes"]


def test_provenance_missing_store_is_404(tmp_path) -> None:
    r = _client(tmp_path).get("/memory/mem_x/provenance")
    assert r.status_code == 404
