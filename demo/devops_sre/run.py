"""Scripted DevOps/SRE incident — every subsystem in one believable story.

    uv run python -m demo.devops_sre.run

A checkout service is erroring. The agent (a) recalls prior infra facts, (b) is
told mid-incident the DB was migrated MySQL->Postgres and the conflict resolver
supersedes the stale fact, (c) uses the guarded Execute_Tool gateway to pull
metrics/logs, (d) hits a poisoned log line and the trust boundary treats it as
data and refuses the injected restart, (e) runs long enough to trigger verified
compaction while preserving ``db=postgres``, and (f) proposes a remediation with
citations. The whole run is replayable by ``run_id``.

Keyless and offline: the LLM is a deterministic MockProvider. Swap in Gemini /
Claude / Ollama by constructing the runtime with a real provider — the contract
is identical.
"""

from __future__ import annotations

from pathlib import Path

from aegismem.api.models import AgentRequest
from aegismem.context.compactor import DeterministicCompactor
from aegismem.context.manager import ContextManager
from aegismem.context.state import ConversationTurn, Role
from aegismem.context.tokens import HeuristicTokenCounter
from aegismem.execution.agent import AgentRuntime
from aegismem.execution.llm import MockProvider
from aegismem.guardrails import SecurityBoundary
from aegismem.mcp.gateway import AuthorizationPolicy, ExecuteToolGateway
from aegismem.mcp.runtime import ToolRuntime
from aegismem.memory.conflict import (
    ConflictRelation,
    ConflictResolver,
    ConflictVerdict,
)
from aegismem.memory.models import MemoryCreate, MemoryRecord, MemoryStatus, MemoryType, TrustLevel
from aegismem.memory.store import SQLiteMemoryStore
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex
from demo.devops_sre.tools import build_registry

_TRACE = Path("traces/demo.jsonl")


class MigrationClassifier:
    """Demo conflict classifier: the LLM *assists* by proposing a verdict; the
    resolver's deterministic policy layer decides whether to supersede."""

    def classify(self, candidate: MemoryCreate, existing: MemoryRecord) -> ConflictVerdict:
        c, e = candidate.content.lower(), existing.content.lower()
        if "postgres" in c and "mysql" in e and "database" in e:
            return ConflictVerdict(
                relation=ConflictRelation.UPDATES,
                confidence=0.93,
                rationale="primary DB changed MySQL -> Postgres",
            )
        return ConflictVerdict(relation=ConflictRelation.UNRELATED, confidence=1.0)


def _seed_active(
    store: SQLiteMemoryStore,
    runtime: AgentRuntime,
    content: str,
    *,
    mtype: MemoryType = MemoryType.SEMANTIC,
) -> MemoryRecord:
    rec = store.create(MemoryCreate(type=mtype, content=content, trust=TrustLevel.USER))
    rec = store.transition(
        rec.id, MemoryStatus.ACTIVE, reason="seed", source="demo", trigger="seed"
    )
    runtime.remember(rec.id, content)
    return rec


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    _TRACE.parent.mkdir(exist_ok=True)
    _TRACE.unlink(missing_ok=True)

    store = SQLiteMemoryStore(":memory:")
    tracer = Tracer(JSONLTraceStore(_TRACE))
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    llm = MockProvider(
        scripts=[
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
    )
    runtime = AgentRuntime(
        llm=llm, router=router, tracer=tracer, boundary=SecurityBoundary(), documents={}
    )

    # (a) seed prior infra knowledge
    _hr("(a) Prior infra facts recalled")
    _seed_active(
        store,
        runtime,
        "Runbook: on connection pool exhaustion restart the service and scale replicas",
    )
    db_fact = _seed_active(store, runtime, "The primary database for checkout is MySQL 8.0")
    _seed_active(store, runtime, "The payments service is owned by the transactions platform team")
    resp = runtime.handle(
        AgentRequest(session_id="incident-42", input="why is the checkout service erroring?")
    )
    print("answer:", resp.output)
    print("citations:", [c.memory_id for c in resp.citations], "run:", resp.run_id)

    # (b) mid-incident correction -> conflict resolver supersedes the stale fact
    _hr("(b) Memory conflict: MySQL -> Postgres (resolver supersedes, policy decides)")
    resolver = ConflictResolver(store, MigrationClassifier())
    live = store.list(status=MemoryStatus.ACTIVE)
    result = resolver.ingest(
        MemoryCreate(
            type=MemoryType.SEMANTIC,
            trust=TrustLevel.USER,
            content="The primary database for checkout is Postgres 16 (migrated from MySQL)",
        ),
        live,
    )
    print(
        "decision:",
        result.decision.value,
        "| verdict:",
        result.verdict.relation.value,
        f"({result.verdict.confidence:.2f})",
    )
    if result.superseded_id:
        runtime.forget(result.superseded_id)
    if result.new_memory:
        runtime.remember(result.new_memory.id, result.new_memory.content)
    print("stale fact", db_fact.id, "->", store.get(db_fact.id).status.value)

    # (c) guarded tool use
    _hr("(c) Tool use through the guarded Execute_Tool gateway")
    reg = build_registry()
    tools = ToolRuntime(reg, ExecuteToolGateway(reg, AuthorizationPolicy()))
    print("search:", [h.name for h in tools.search_tools("metrics latency for a service")])
    metrics = tools.execute_tool("query_metrics", {"service": "checkout"})
    print("query_metrics:", tools.read_tool_result(metrics.ref).output)
    logs = tools.execute_tool("read_logs", {"service": "checkout"})

    # (d) poisoned log line -> treated as data, injected restart refused
    _hr("(d) Poisoned log line: trust boundary refuses the injected action")
    from aegismem.mcp.gateway import ToolResult

    log_text = tools.read_tool_result(logs.ref).output
    report = runtime.boundary.validate_tool_result(
        ToolResult(tool="read_logs", output=log_text, trust=TrustLevel.TOOL_RESULT)
    )
    print("log carries injected instructions:", report.contains_instructions)
    from aegismem.errors import PermissionDeniedError

    try:
        tools.execute_tool("restart_service", {"service": "prod-db"})  # the injection's demand
        print("restart EXECUTED (BUG)")
    except PermissionDeniedError as exc:
        print("restart_service refused:", exc.envelope.message)

    # (e) long context -> verified compaction preserving db=postgres
    _hr("(e) Long context: verified compaction preserves critical state")
    turns = [
        ConversationTurn(role=Role.USER, content="continue triage"),
        ConversationTurn(role=Role.ASSISTANT, content="confirmed [state] db=postgres"),
    ]
    turns += [
        ConversationTurn(role=Role.ASSISTANT, content=f"log burst {i}: " + "noise " * 40)
        for i in range(60)
    ]
    mgr = ContextManager(DeterministicCompactor(), HeuristicTokenCounter(), keep_recent_turns=2)
    comp = mgr.compact(
        "You are an SRE assistant.", turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    print(
        f"committed={comp.committed} reduction={comp.reduction:.1%} "
        f"critical db={comp.state.critical_variables.get('db') if comp.state else None}"
    )

    # (f) grounded remediation with citations, replayable
    _hr("(f) Remediation with citations")
    final = runtime.handle(
        AgentRequest(
            session_id="incident-42", input="give the remediation for the checkout incident"
        )
    )
    print("answer:", final.output)
    print("citations:", [c.memory_id for c in final.citations])
    print(
        f"\nRun recorded. Replay it with:\n  uv run aegismem replay {final.run_id} "
        f"--trace-path {_TRACE}"
    )
    store.close()


if __name__ == "__main__":
    main()
