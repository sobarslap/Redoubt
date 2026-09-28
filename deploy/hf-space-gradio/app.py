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
# 🛡️ AegisMem
### A production-grade agent runtime — framework-free, pure Python

**Lifecycle-managed memory** · **hybrid retrieval** · **prompt-injection trust
boundary** · **guarded tool execution** · **deterministic replay**.

This live demo has four DevOps/SRE incident memories seeded. Ask a **grounded**
question (about the checkout incident) for a cited answer — or an **ungrounded**
one (*"what is the capital of France?"*) and watch it refuse rather than
hallucinate.

[⌥ Source on GitHub →](https://github.com/sobarslap/Redoubt)
"""

_EXAMPLES = [
    "why is the checkout service erroring?",
    "what database does checkout use?",
    "give the remediation for the incident",
    "what is the capital of France?",
]

# Design direction from the ui-ux-pro-max skill's database:
# palette "Developer Tool / IDE" (slate #0F172A + blue #2563EB/#3B82F6),
# type pairing "Tech Startup" (Space Grotesk headings / DM Sans body, JetBrains
# Mono for identifiers). Applied as a proper Gradio theme so it holds on the Space.
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Space Grotesk"), gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0F172A",
    body_text_color="#F1F5F9",
    background_fill_primary="#1E293B",
    background_fill_secondary="#0F172A",
    block_background_fill="#1E293B",
    block_border_color="#334155",
    block_label_text_color="#94A3B8",
    border_color_primary="#334155",
    input_background_fill="#0F172A",
    input_border_color="#334155",
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_hover="#3B82F6",
    button_primary_text_color="#FFFFFF",
    body_text_color_subdued="#94A3B8",
    link_text_color="#60A5FA",
)

_CSS = """
.gradio-container {max-width: 960px !important; margin: 0 auto !important;}
#answer_box {border-left: 3px solid #2563EB; padding-left: 14px;}
footer {display: none !important;}
"""

with gr.Blocks(title="AegisMem — agent runtime", theme=_THEME, css=_CSS) as demo:
    gr.Markdown(_INTRO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            q = gr.Textbox(
                label="Ask the runtime",
                placeholder="why is the checkout service erroring?",
                value="why is the checkout service erroring?",
                lines=2,
            )
            go = gr.Button("Ask AegisMem", variant="primary", size="lg")
            gr.Examples(_EXAMPLES, inputs=q, label="Try one")
        with gr.Column(scale=4):
            answer = gr.Markdown(elem_id="answer_box")
            evidence = gr.Markdown()

    go.click(ask, inputs=q, outputs=[answer, evidence])
    q.submit(ask, inputs=q, outputs=[answer, evidence])

if __name__ == "__main__":
    demo.launch()
