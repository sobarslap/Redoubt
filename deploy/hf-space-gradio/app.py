"""Hugging Face Space (Gradio) — interactive AegisMem demo UI.

A small, framework-honest front end over the AegisMem runtime. It seeds the
DevOps/SRE incident memories, then lets a visitor ask questions and see:

* the grounded answer (or a refusal when nothing can be cited),
* the exact memories it grounded on, with relevance scores,
* the run_id / trace_id proving the run was recorded and is replayable.

The retrieval path (BM25 + hashing embedder + overlap reranker) is pure Python,
so the Space needs no ML/torch dependencies. Set GEMINI_API_KEY or
ANTHROPIC_API_KEY as a Space secret to answer over the same grounded context with
a real model instead of the keyless mock.
"""

from __future__ import annotations

import gradio as gr

from aegismem.api.models import AgentRequest
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

_MEMORIES: list[tuple[str, str]] = [
    ("mem_runbook", "Runbook: on connection pool exhaustion restart the service and scale replicas"),
    ("mem_db", "The primary database for checkout is Postgres 16 (migrated from MySQL)"),
    ("mem_metrics", "checkout p95_latency=1240ms error_rate=7.2% conns=198/200 (pool near exhaustion)"),
    ("mem_owner", "The payments service is owned by the transactions platform team"),
]


def _build_runtime() -> AgentRuntime:
    import tempfile

    trace_store = JSONLTraceStore(f"{tempfile.gettempdir()}/aegismem-ui.jsonl")
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    llm = build_client("workhorse")
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
        llm=llm, router=router, tracer=Tracer(trace_store), boundary=SecurityBoundary(), documents={}
    )
    for mem_id, content in _MEMORIES:
        runtime.remember(mem_id, content)
    return runtime


RUNTIME = _build_runtime()


def ask(question: str) -> tuple[str, str]:
    """Run one query through the full pipeline; return (answer_md, evidence_md)."""
    question = (question or "").strip()
    if not question:
        return "_Ask something to see a grounded answer._", ""
    resp = RUNTIME.handle(AgentRequest(session_id="demo", input=question))

    answer_md = f"### Answer\n\n{resp.output}\n"
    if not resp.citations:
        answer_md += (
            "\n> ⚠️ **No grounding found.** The runtime refuses to fabricate — it only "
            "answers from memories it can cite. Try a question about the checkout incident."
        )

    lines = [
        f"**status** `{resp.status.value}` · **run** `{resp.run_id}` · "
        f"**trace** `{resp.trace_id}`",
        f"**tokens** in {resp.usage.input_tokens} / out {resp.usage.output_tokens} · "
        f"**tool calls** {resp.usage.tool_calls}",
        "",
    ]
    if resp.citations:
        lines.append("### Grounded on")
        for c in resp.citations:
            lines.append(f"- `{c.memory_id}`  ·  score **{c.score:.3f}**  \n  {c.snippet}")
    evidence_md = "\n".join(lines)
    return answer_md, evidence_md


_INTRO = """
# 🛡️ AegisMem — a production-grade agent runtime

Framework-free, pure-Python: **lifecycle-managed memory**, **hybrid retrieval**,
a **prompt-injection trust boundary**, **guarded tool execution**, and
**deterministic replay**. This demo has four DevOps/SRE incident memories seeded.

Ask a **grounded** question (about the checkout incident) to get a cited answer —
or an **ungrounded** one (e.g. *"what is the capital of France?"*) to watch it
refuse rather than hallucinate.

[Source on GitHub](https://github.com/sobarslap/Redoubt)
"""

_EXAMPLES = [
    "why is the checkout service erroring?",
    "what database does checkout use?",
    "give the remediation for the incident",
    "what is the capital of France?",
]

with gr.Blocks(title="AegisMem", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            q = gr.Textbox(
                label="Question",
                placeholder="why is the checkout service erroring?",
                value="why is the checkout service erroring?",
                lines=2,
            )
            go = gr.Button("Ask AegisMem", variant="primary")
            gr.Examples(_EXAMPLES, inputs=q, label="Try one")
        with gr.Column(scale=4):
            answer = gr.Markdown(label="Answer")
            evidence = gr.Markdown(label="Evidence")

    go.click(ask, inputs=q, outputs=[answer, evidence])
    q.submit(ask, inputs=q, outputs=[answer, evidence])

if __name__ == "__main__":
    demo.launch()
