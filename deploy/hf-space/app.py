"""Hugging Face Space entry point — AegisMem HTTP API with seeded demo memories.

This is the public demo face of AegisMem. It installs the `aegismem` package
(core deps only) and serves the same FastAPI service the project ships, but with
the DevOps/SRE incident memories pre-seeded into the retrieval router so a `/runs`
call returns a full, cited answer instead of the empty-store acknowledgment.

The whole retrieval path used here (BM25 + hashing embedder + overlap reranker) is
pure Python, so the image needs no ML/torch dependencies. Set GEMINI_API_KEY or
ANTHROPIC_API_KEY as a Space secret to answer over the same grounded context with a
real model instead of the keyless mock.
"""

from __future__ import annotations

import os
import tempfile

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

# The facts a real ingestion pipeline would have written; seeded so the live demo
# has something to ground on and cite.
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
    trace_path = os.path.join(tempfile.gettempdir(), "aegismem-service.jsonl")
    trace_store = JSONLTraceStore(trace_path)
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)

    llm = build_client("workhorse")
    if isinstance(llm, MockProvider):
        # Keyless -> mock: scripted grounded answers so it responds like a real model
        # once the memories are retrieved. With a provider key set as a Space secret,
        # this branch is skipped and the live model answers over the same context.
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
    # Hugging Face Spaces routes to the port declared in README.md (app_port: 7860).
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
