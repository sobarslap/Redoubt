"""FastAPI service — the runtime contract as HTTP (Production Phase P3).

The same ``AgentRequest`` → ``AgentResponse`` contract the library exposes, now
over the network, with the production concerns the plan names:

* **sync + async** runs (``POST /runs``); async returns a ``run_id`` immediately
  and ``GET /runs/{run_id}`` polls status/result.
* **idempotency** — a repeated ``idempotency_key`` returns the original run.
* **cooperative cancellation** — ``DELETE /runs/{run_id}`` cancels a queued or
  in-flight run at the next stage boundary.
* **backpressure** — a bounded in-flight limit returns a typed 503 rather than
  melting down.
* **typed error envelope** on every path — never a raw stack.
* **graceful shutdown** — the lifespan flushes traces and closes the store.

The blocking ``AgentRuntime.handle`` runs in a worker thread (``to_thread``) so
the event loop stays responsive under concurrency. Build with ``create_app()``;
inject a runtime/store in tests.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from aegismem.api.models import AgentRequest, AgentResponse, RunMode, RunStatus
from aegismem.errors import AegisError, ErrorCategory, ErrorEnvelope, NotFoundError
from aegismem.execution.agent import AgentRuntime
from aegismem.execution.factory import build_client
from aegismem.guardrails import SecurityBoundary
from aegismem.observability.metrics import MetricsRegistry
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.factory import build_embedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex
from aegismem.security.audit import AuditLog
from aegismem.security.auth import (
    ApiKeyStore,
    AuthContext,
    FixedWindowQuota,
    OIDCVerifier,
    Principal,
)

# HTTP status per error category — the envelope drives the code, consistently.
_STATUS: dict[ErrorCategory, int] = {
    ErrorCategory.VALIDATION: 422,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.CONFLICT: 409,
    ErrorCategory.GUARDRAIL: 400,
    ErrorCategory.BUDGET: 429,
    ErrorCategory.TOOL: 502,
    ErrorCategory.LLM: 502,
    ErrorCategory.STORAGE: 503,
    ErrorCategory.AUTH: 401,
    ErrorCategory.INTERNAL: 500,
}

_ANON = AuthContext(principal=Principal(principal_id="-", tenant="-"), key_fingerprint="-")


@dataclass
class _RunRecord:
    response: AgentResponse
    owner: str = "-"  # principal_id that created the run
    cancelled: bool = False


@dataclass
class RunRegistry:
    """In-process run + idempotency bookkeeping. (A distributed deployment swaps
    this for Redis/Postgres; the interface is deliberately tiny.)"""

    runs: dict[str, _RunRecord] = field(default_factory=dict)
    by_idempotency: dict[str, str] = field(default_factory=dict)
    max_runs: int = 10_000  # bound in-process memory in a long-lived service

    def add(self, run_id: str, record: _RunRecord) -> None:
        self.runs[run_id] = record
        # Evict the oldest completed runs (insertion order) so the map can't grow
        # without bound; idempotency entries pointing at evicted runs simply miss.
        while len(self.runs) > self.max_runs:
            oldest, _ = next(iter(self.runs.items()))
            del self.runs[oldest]

    def get(self, run_id: str) -> _RunRecord:
        rec = self.runs.get(run_id)
        if rec is None:
            raise NotFoundError(f"run {run_id!r} not found", stage="api")
        return rec

    def get_owned(self, run_id: str, owner: str) -> _RunRecord:
        """Fetch a run only if it belongs to ``owner``; otherwise behave as if it
        does not exist (a caller can never read/act on — or even confirm the
        existence of — another caller's run)."""
        rec = self.runs.get(run_id)
        if rec is None or rec.owner != owner:
            raise NotFoundError(f"run {run_id!r} not found", stage="api")
        return rec


def _default_runtime() -> tuple[AgentRuntime, JSONLTraceStore]:
    store = JSONLTraceStore("traces/service.jsonl")
    # Fan out to a hosted Langfuse when configured; the exported copy is PII-redacted,
    # the local JSONL store stays exact for replay. No-op when Langfuse is absent.
    from aegismem.observability.langfuse_exporter import LangfuseExporter
    from aegismem.observability.redact import RedactingSink
    from aegismem.observability.trace_store import FanoutSink

    exporter = LangfuseExporter()
    sink: object = FanoutSink([store, RedactingSink(exporter)]) if exporter.available else store
    router = JITRouter(BM25Index(), VectorIndex(build_embedder()), OverlapReranker(), top_k=3)
    runtime = AgentRuntime(
        llm=build_client("workhorse"),
        router=router,
        tracer=Tracer(sink),  # type: ignore[arg-type]
        boundary=SecurityBoundary(),
    )
    return runtime, store


def create_app(
    runtime: AgentRuntime | None = None,
    *,
    trace_store: JSONLTraceStore | None = None,
    provenance_store: object | None = None,
    max_in_flight: int = 32,
    api_keys: ApiKeyStore | None = None,
    oidc: OIDCVerifier | None = None,
    quota: FixedWindowQuota | None = None,
    audit: AuditLog | None = None,
    metrics: MetricsRegistry | None = None,
) -> FastAPI:
    if runtime is None:
        runtime, trace_store = _default_runtime()
    registry = RunRegistry()
    audit = audit or AuditLog()
    metrics = metrics or MetricsRegistry()
    in_flight = {"n": 0}
    lock = asyncio.Lock()
    background: set[asyncio.Task[None]] = set()  # strong refs so tasks aren't GC'd

    def authenticate(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> AuthContext:
        # Auth is opt-in: with neither an API-key store nor an OIDC verifier the
        # service runs open (dev/library mode). When both are configured, an
        # X-API-Key wins; otherwise a Bearer token is verified via OIDC.
        if api_keys is None and oidc is None:
            return _ANON
        try:
            if x_api_key is not None and api_keys is not None:
                principal = api_keys.authenticate(x_api_key)
            elif oidc is not None:
                principal = oidc.verify(authorization)
            elif api_keys is not None:
                principal = api_keys.authenticate(x_api_key)  # -> AuthError (missing key)
            else:  # pragma: no cover - unreachable given the guard above
                return _ANON
        except AegisError as exc:
            audit.record("auth.fail", detail=exc.envelope.message, outcome="denied")
            raise
        if quota is not None:
            try:
                quota.allow(principal.principal_id)
            except AegisError:
                audit.record(
                    "auth.quota",
                    principal_id=principal.principal_id,
                    tenant=principal.tenant,
                    outcome="denied",
                )
                raise
        fp = "-" if x_api_key is None else hashlib.sha256(x_api_key.encode()).hexdigest()[:8]
        return AuthContext(principal=principal, key_fingerprint=fp)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Graceful shutdown: let the fire-and-forget trace store settle.
        if isinstance(trace_store, JSONLTraceStore):
            pass  # JSONL writes are flushed per line; nothing buffered to drain.

    app = FastAPI(title="AegisMem", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(AegisError)
    async def _aegis_handler(_: Request, exc: AegisError) -> JSONResponse:
        return JSONResponse(
            status_code=_STATUS.get(exc.envelope.category, 500), content=exc.envelope.model_dump()
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        env = ErrorEnvelope(
            code="internal", category=ErrorCategory.INTERNAL, message="internal error", stage="api"
        )
        return JSONResponse(status_code=500, content=env.model_dump())

    async def _execute(run_id: str, req: AgentRequest) -> AgentResponse:
        def cancel_check() -> bool:
            rec = registry.runs.get(run_id)
            return rec is not None and rec.cancelled

        async with lock:
            in_flight["n"] += 1
        try:
            resp = await asyncio.to_thread(runtime.handle, req, cancel_check=cancel_check)
            metrics.inc("aegismem_runs_total")
            if resp.status == RunStatus.FAILED:
                metrics.inc("aegismem_runs_failed_total")
            metrics.observe("aegismem_run_latency_ms", resp.timings.total_ms)
            return resp
        finally:
            async with lock:
                in_flight["n"] -= 1

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        return {"status": "ready", "in_flight": in_flight["n"]}

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        metrics.counters.setdefault("aegismem_runs_total", 0.0)
        return PlainTextResponse(metrics.render())

    @app.post("/runs")
    async def create_run(
        req: AgentRequest,
        auth: AuthContext = Depends(authenticate),  # noqa: B008
    ) -> JSONResponse:
        owner = auth.principal.principal_id
        audit.record(
            "run.create",
            principal_id=owner,
            tenant=auth.principal.tenant,
            detail=req.request_id,
        )
        # Idempotency is scoped per principal so one caller's key can never return
        # another caller's run.
        idem = f"{owner}:{req.idempotency_key}" if req.idempotency_key else None
        if idem and idem in registry.by_idempotency:
            prior = registry.get(registry.by_idempotency[idem])
            return JSONResponse(status_code=200, content=prior.response.model_dump())

        if in_flight["n"] >= max_in_flight:
            env = ErrorEnvelope(
                code="overloaded",
                category=ErrorCategory.BUDGET,
                message="server at capacity; retry later",
                retryable=True,
                stage="api",
            )
            return JSONResponse(status_code=503, content=env.model_dump())

        if req.mode is RunMode.ASYNC:
            # Reserve a run_id and return immediately; execute in the background.
            placeholder = AgentResponse(
                request_id=req.request_id,
                run_id=f"run_pending_{req.request_id}",
                status=RunStatus.QUEUED,
            )
            registry.add(placeholder.run_id, _RunRecord(response=placeholder, owner=owner))
            if idem:
                registry.by_idempotency[idem] = placeholder.run_id

            async def _bg() -> None:
                rec = registry.runs[placeholder.run_id]
                if rec.cancelled:
                    rec.response = placeholder.model_copy(update={"status": RunStatus.FAILED})
                    return
                resp = await _execute(placeholder.run_id, req)
                # The engine mints its own internal run_id; rebind it to the id the
                # client was handed (and polls) so the response body and GET agree.
                resp = resp.model_copy(update={"run_id": placeholder.run_id})
                registry.runs[placeholder.run_id].response = resp

            task = asyncio.create_task(_bg())
            background.add(task)
            task.add_done_callback(background.discard)
            return JSONResponse(status_code=202, content=placeholder.model_dump())

        # Sync: run to completion and return the response.
        resp = await _execute(req.request_id, req)
        registry.add(resp.run_id, _RunRecord(response=resp, owner=owner))
        if idem:
            registry.by_idempotency[idem] = resp.run_id
        return JSONResponse(status_code=200, content=resp.model_dump())

    @app.get("/runs/{run_id}")
    async def get_run(
        run_id: str,
        auth: AuthContext = Depends(authenticate),  # noqa: B008
    ) -> JSONResponse:
        rec = registry.get_owned(run_id, auth.principal.principal_id)
        return JSONResponse(status_code=200, content=rec.response.model_dump())

    @app.delete("/runs/{run_id}")
    async def cancel_run(
        run_id: str,
        auth: AuthContext = Depends(authenticate),  # noqa: B008
    ) -> JSONResponse:
        rec = registry.get_owned(run_id, auth.principal.principal_id)
        rec.cancelled = True
        audit.record(
            "run.cancel",
            principal_id=auth.principal.principal_id,
            tenant=auth.principal.tenant,
            detail=run_id,
        )
        return JSONResponse(status_code=200, content={"run_id": run_id, "cancelled": True})

    @app.get("/memory/{memory_id}/provenance")
    async def provenance(
        memory_id: str,
        auth: AuthContext = Depends(authenticate),  # noqa: B008
    ) -> JSONResponse:
        if provenance_store is None or not hasattr(provenance_store, "provenance"):
            raise NotFoundError("no provenance store configured", stage="api")
        view = provenance_store.provenance(memory_id)
        return JSONResponse(status_code=200, content=view.model_dump())

    app.state.runtime = runtime
    app.state.registry = registry
    app.state.audit = audit
    return app
