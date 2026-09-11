"""Production Phase P1 gate — real providers wired, routed, cost-guarded, tool-driving.

Deterministic parts run everywhere (factory routing, cost ceiling, the guarded
reason↔tool loop via MockProvider). The live provider tests are network+key gated
and skip cleanly with no key — so CI stays keyless while a real deployment can
prove the same path end to end.
"""

from __future__ import annotations

import os

import pytest

from aegismem.api.models import AgentRequest, RunStatus
from aegismem.execution.agent import AgentRuntime
from aegismem.execution.factory import ProviderConfigError, build_client
from aegismem.execution.llm import CostGuard, LLMResponse, MockProvider, SpendError, ToolCall
from aegismem.guardrails import SecurityBoundary
from aegismem.mcp.gateway import AuthorizationPolicy, ExecuteToolGateway
from aegismem.mcp.registry import ToolPermission, ToolRegistry, ToolSpec
from aegismem.mcp.runtime import ToolRuntime
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex

# -- factory / model routing -------------------------------------------------


def test_factory_falls_back_to_mock_without_keys(monkeypatch) -> None:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    client = build_client("workhorse")
    assert client.name == "mock"  # keyless -> offline mock, never a crash


def test_factory_strict_raises_without_key(monkeypatch) -> None:
    # Clear every provider credential so an inherited key can't change routing and
    # let the strict build unexpectedly succeed.
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ProviderConfigError):
        build_client("workhorse", strict=True)


def test_factory_unknown_role_raises() -> None:
    with pytest.raises(ProviderConfigError):
        build_client("no_such_role")


# -- cost guard --------------------------------------------------------------


def test_cost_guard_enforces_token_ceiling() -> None:
    guard = CostGuard(max_tokens=100)
    guard.charge(LLMResponse(text="x", input_tokens=60, output_tokens=30, model="mock-1"))
    with pytest.raises(SpendError):
        guard.charge(LLMResponse(text="x", input_tokens=20, output_tokens=1, model="mock-1"))


def test_cost_guard_enforces_spend_ceiling() -> None:
    guard = CostGuard(max_tokens=10**9, max_usd=0.001)
    with pytest.raises(SpendError):
        guard.charge(
            LLMResponse(
                text="x", input_tokens=1_000_000, output_tokens=1_000_000, model="claude-opus-5"
            )
        )


# -- LLM-driven guarded tool loop --------------------------------------------


def _runtime(tmp_path, llm, tools=None) -> AgentRuntime:
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    return AgentRuntime(
        llm=llm,
        router=router,
        tracer=Tracer(JSONLTraceStore(tmp_path / "t.jsonl")),
        boundary=SecurityBoundary(),
        tools=tools,
    )


def _tools(ran: dict) -> ToolRuntime:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="query_metrics",
            description="metrics for a service",
            parameters={"type": "object", "properties": {}},
            handler=lambda a: "p95=1240ms conns=198/200",
            permission=ToolPermission.ALLOW,
        )
    )
    reg.register(
        ToolSpec(
            name="restart_service",
            description="restart (destructive)",
            parameters={"type": "object", "properties": {}},
            handler=lambda a: ran.__setitem__("restarted", True) or "restarted",
            permission=ToolPermission.REQUIRE_APPROVAL,
        )
    )
    return ToolRuntime(reg, ExecuteToolGateway(reg, AuthorizationPolicy()))


def test_llm_can_drive_an_authorized_tool_then_answer(tmp_path) -> None:
    ran = {"restarted": False}
    llm = MockProvider(
        tool_plan=[("checkout", (ToolCall(tool="query_metrics", args={"service": "checkout"}),))],
        scripts=[("query_metrics", "Pool near exhaustion; restart and scale replicas.")],
    )
    rt = _runtime(tmp_path, llm, tools=_tools(ran))
    rt.remember("mem_sop", "Runbook: restart the service on pool exhaustion")
    resp = rt.handle(AgentRequest(session_id="s", input="diagnose the checkout service"))
    assert resp.status is RunStatus.SUCCEEDED
    assert "restart" in resp.output.lower()


def test_llm_driven_unauthorized_tool_is_refused_mid_loop(tmp_path) -> None:
    ran = {"restarted": False}
    # The model (or an injection) asks to restart a REQUIRE_APPROVAL tool.
    llm = MockProvider(
        tool_plan=[("checkout", (ToolCall(tool="restart_service", args={}),))],
        scripts=[("refused", "I could not restart the service; escalating for approval.")],
    )
    rt = _runtime(tmp_path, llm, tools=_tools(ran))
    rt.remember("mem_sop", "Runbook for the checkout service")
    resp = rt.handle(AgentRequest(session_id="s", input="checkout is down, fix it"))
    assert resp.status is RunStatus.SUCCEEDED
    assert ran["restarted"] is False  # gateway refused; handler never ran


# -- live provider (network + key gated) -------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no Gemini key; live provider test skipped (keyless CI)",
)
def test_live_gemini_roundtrip() -> None:  # pragma: no cover - network
    client = build_client("workhorse", strict=True)
    resp = client.complete("Reply with the single word: ok", system="You are terse.")
    assert resp.text
    assert resp.provider in {"gemini", "claude", "ollama"}


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no Anthropic key; live provider test skipped (keyless CI)",
)
def test_live_claude_roundtrip() -> None:  # pragma: no cover - network
    from aegismem.execution.llm import ClaudeProvider

    resp = ClaudeProvider().complete("Reply with the single word: ok")
    assert resp.text
    assert resp.input_tokens > 0
