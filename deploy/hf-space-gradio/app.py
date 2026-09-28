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

from aegismem.api.models import AgentRequest, RunStatus
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
            (
                "exhaustion",
                "Connection-pool exhaustion: per the runbook, restart the service and scale "
                "replicas. [cites grounded memory]",
            ),
        ]
    banner = (
        "real Gemini model (grounded answers)"
        if not isinstance(llm, MockProvider)
        else "OFFLINE MOCK (scripted answers) — set GEMINI_API_KEY for real answers"
    )
    print("=" * 70)
    print(f"  AegisMem demo  |  LLM provider: {llm.name}  ->  {banner}")
    print("=" * 70)
    runtime = AgentRuntime(
        llm=llm,
        router=router,
        tracer=Tracer(trace_store),
        boundary=SecurityBoundary(),
        documents={},
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

    # Never let an exception leave the UI spinning with nothing shown.
    try:
        resp = RUNTIME.handle(AgentRequest(session_id="demo", input=question))
    except Exception as exc:  # surface any failure to the user, never hang
        return (
            "### ⚠️ Something went wrong\n\nThe request failed before it finished. "
            "This is usually a transient model issue — please try again.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        ), ""

    ids_line = (
        f"**status** `{resp.status.value}` · **run** `{resp.run_id}` · **trace** `{resp.trace_id}`"
    )
    tokens_line = (
        f"**tokens** in {resp.usage.input_tokens} / out {resp.usage.output_tokens} · "
        f"**tool calls** {resp.usage.tool_calls}"
    )

    # A failed run is NOT a refusal — the model call errored (rate-limit / overload /
    # timeout). Say so explicitly instead of showing an empty "refusal" or hanging.
    if resp.status == RunStatus.FAILED:
        detail = resp.error.message if resp.error else "unknown error"
        low = detail.lower()
        if "429" in low or "resource_exhausted" in low or "quota" in low:
            note = (
                "You've hit the **free-tier rate limit** (Gemini allows only a few "
                "requests per minute). Wait ~30-60 seconds and try again - this is a "
                "quota limit, not a bug. (Run keyless to use the offline mock with no limit.)"
            )
        else:
            note = (
                "The language model didn't return a response — usually a transient "
                "overload. Please try again in a moment."
            )
        answer_md = f"### ⚠️ Model call failed\n\n{note}\n\n`{detail}`"
        return answer_md, "\n".join([ids_line, tokens_line])

    # Succeeded with no citations means retrieval returned EMPTY MEMORY — nothing
    # cleared the relevance gate. Show that as an explicit refusal, not a vague answer.
    if not resp.citations:
        answer_md = (
            "### Refused — no grounded memory\n\n"
            "Nothing in memory is relevant to that question, so the runtime **won't "
            "answer** — it refuses to fabricate rather than guess.\n"
        )
        evidence_md = "\n".join(
            [ids_line, tokens_line, "", "**Grounded on**", "_EMPTY MEMORY — no memory was cited._"]
        )
        return answer_md, evidence_md

    answer_md = f"### Answer\n\n{resp.output}\n"
    lines = [ids_line, tokens_line, "", "### Grounded on"]
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
    "how do we fix connection pool exhaustion?",
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
