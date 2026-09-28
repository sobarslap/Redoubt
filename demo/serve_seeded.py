"""Serve the AegisMem HTTP API with the DevOps/SRE incident memories pre-seeded.

The bare `aegismem.api.app:create_app` service starts with an empty memory store,
so `/runs` returns the grounded-mode acknowledgment (nothing to cite). This
launcher builds the same runtime, seeds the checkout-incident facts into the
retrieval router, and serves it — so a `/runs` call against the running server
returns the full, cited answer exactly like the offline demo.

Run it (instead of the bare uvicorn command):

    uv run python -m demo.serve_seeded

then open http://127.0.0.1:8000/docs and POST /runs with:

    {"session_id": "s", "input": "why is the checkout service erroring?"}
"""

from __future__ import annotations

import uvicorn

from aegismem.api.app import create_app
from aegismem.execution.agent import AgentRuntime
from aegismem.execution.factory import build_client
from aegismem.execution.llm import MockProvider
from aegismem.guardrails import SecurityBoundary
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex

# The facts a real ingestion pipeline would have written; seeded here so the live
# service has something to ground on and cite.
_MEMORIES: list[tuple[str, str]] = [
    (
        "mem_runbook",
        "Runbook: on connection pool exhaustion restart the service and scale replicas",
    ),
    ("mem_db", "The primary database for checkout is Postgres 16 (migrated from MySQL)"),
    (
        "mem_metrics",
        "checkout p95_latency=1240ms error_rate=7.2% conns=198/200 (pool near exhaustion)",
    ),
    ("mem_owner", "The payments service is owned by the transactions platform team"),
]


def build_seeded_app():
    trace_store = JSONLTraceStore("traces/service.jsonl")
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)

    llm = build_client("workhorse")
    # Keyless -> MockProvider: give it the scripted grounded answers so it responds
    # like a real model once the memories are retrieved. With a real key set
    # (GEMINI_API_KEY / ANTHROPIC_API_KEY) this branch is skipped and the live model
    # answers over the same seeded, grounded context.
    if isinstance(llm, MockProvider):
        llm.scripts = [
            (
                "remediation",
                "Remediation: primary DB is Postgres (migrated from MySQL). Pool is near "
                "exhaustion (198/200); restart checkout and scale replicas per the runbook.",
            ),
            (
                "checkout",
                "The checkout service errors trace to connection-pool exhaustion. "
                "Per the runbook, restart the service and scale replicas. [cites grounded memory]",
            ),
        ]

    runtime = AgentRuntime(
        llm=llm,
        router=router,
        tracer=Tracer(trace_store),
        boundary=SecurityBoundary(),
        documents={},
    )
    for mem_id, content in _MEMORIES:
        runtime.remember(mem_id, content)

    print(f"LLM provider: {llm.name}  |  seeded {len(_MEMORIES)} memories")
    return create_app(runtime=runtime, trace_store=trace_store)


app = build_seeded_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
