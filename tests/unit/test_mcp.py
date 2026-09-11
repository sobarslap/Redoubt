"""Phase 5 gate: progressive MCP tool discovery + the discovery/authorization split.

The gate: a discovered tool never auto-becomes an authorized tool. The invariant:
unauthorized_tool_exec = 0 — the handler runs only after every guard passes.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from aegismem.errors import (
    ArgumentValidationError,
    BudgetExceededError,
    PermissionDeniedError,
    ToolTimeoutError,
)
from aegismem.mcp.gateway import AuthorizationPolicy, ExecuteToolGateway, RunBudget
from aegismem.mcp.registry import ToolPermission, ToolRegistry, ToolSpec
from aegismem.mcp.runtime import ToolRuntime


class QueryMetricsArgs(BaseModel):
    service: str
    window_min: int = 5


def _spec(
    name: str,
    description: str,
    handler,
    *,
    permission: ToolPermission = ToolPermission.REQUIRE_APPROVAL,
    args_model: type[BaseModel] | None = None,
    timeout_s: float = 5.0,
    max_exec: int = 3,
    result_cap: int = 8000,
    operations: frozenset[str] = frozenset(),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
        handler=handler,
        args_model=args_model,
        permission=permission,
        timeout_s=timeout_s,
        max_executions_per_run=max_exec,
        result_size_cap=result_cap,
        operations=operations,
    )


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        _spec(
            "query_metrics",
            "Query time-series metrics for a service over a window",
            lambda a: f"p95={a.get('window_min', 5) * 10}ms for {a['service']}",
            permission=ToolPermission.ALLOW,
            args_model=QueryMetricsArgs,
        )
    )
    reg.register(
        _spec(
            "read_logs",
            "Read recent log lines for a service",
            lambda a: "log line 1\nlog line 2",
            permission=ToolPermission.ALLOW,
        )
    )
    reg.register(
        _spec(
            "restart_service",
            "Restart a running service (destructive)",
            lambda a: "restarted",
            permission=ToolPermission.REQUIRE_APPROVAL,
        )
    )
    return reg


# -- discovery ---------------------------------------------------------------


def test_search_finds_tools_by_description() -> None:
    reg = _registry()
    hits = reg.search("metrics latency", top_k=2)
    assert hits[0].name == "query_metrics"


def test_progressive_injection_is_small_vs_catalog() -> None:
    reg = _registry()
    injected = reg.inject_schemas("metrics", top_k=1)
    injected_chars = sum(len(str(s)) for s in injected)
    assert injected_chars < reg.catalog_schema_chars()


# -- the gate: discovery is not authorization --------------------------------


def test_discovered_tool_is_not_auto_authorized() -> None:
    reg = _registry()
    ran = {"restart": False}
    reg.register(
        _spec(
            "restart_service",
            "Restart a running service (destructive)",
            lambda a: ran.__setitem__("restart", True) or "restarted",
            permission=ToolPermission.REQUIRE_APPROVAL,
        )
    )
    runtime = ToolRuntime(reg, ExecuteToolGateway(reg, AuthorizationPolicy()))

    # Discovery surfaces it...
    assert any(h.name == "restart_service" for h in runtime.search_tools("restart service"))
    # ...but executing it is refused, and the handler never runs.
    with pytest.raises(PermissionDeniedError):
        runtime.execute_tool("restart_service", {})
    assert ran["restart"] is False


def test_deny_list_overrides_default_allow() -> None:
    reg = _registry()
    policy = AuthorizationPolicy(deny=frozenset({"read_logs"}))
    gw = ExecuteToolGateway(reg, policy)
    with pytest.raises(PermissionDeniedError):
        gw.execute("read_logs", {})
    assert gw.executions == 0


def test_allow_list_authorizes_a_require_approval_tool() -> None:
    reg = _registry()
    policy = AuthorizationPolicy(allow=frozenset({"restart_service"}))
    gw = ExecuteToolGateway(reg, policy)
    result = gw.execute("restart_service", {})
    assert result.output == "restarted"
    assert gw.executions == 1


# -- guards ------------------------------------------------------------------


def test_argument_validation_blocks_bad_args() -> None:
    reg = _registry()
    gw = ExecuteToolGateway(reg, AuthorizationPolicy())
    with pytest.raises(ArgumentValidationError):
        gw.execute("query_metrics", {"window_min": "not-an-int"})  # missing service too
    assert gw.executions == 0


def test_valid_args_execute_and_return_trusted_result() -> None:
    reg = _registry()
    gw = ExecuteToolGateway(reg, AuthorizationPolicy())
    result = gw.execute("query_metrics", {"service": "checkout", "window_min": 3})
    assert "p95=30ms" in result.output
    assert result.trust.value == "tool_result"


def test_per_tool_budget_is_enforced() -> None:
    reg = ToolRegistry()
    reg.register(
        _spec(
            "ping",
            "ping a host",
            lambda a: "pong",
            permission=ToolPermission.ALLOW,
            max_exec=2,
        )
    )
    gw = ExecuteToolGateway(reg, AuthorizationPolicy())
    gw.execute("ping", {})
    gw.execute("ping", {})
    with pytest.raises(BudgetExceededError):
        gw.execute("ping", {})
    assert gw.executions == 2


def test_run_call_budget_is_enforced() -> None:
    reg = ToolRegistry()
    reg.register(
        _spec("ping", "ping", lambda a: "pong", permission=ToolPermission.ALLOW, max_exec=100)
    )
    gw = ExecuteToolGateway(reg, AuthorizationPolicy(), RunBudget(max_tool_calls=1))
    gw.execute("ping", {})
    with pytest.raises(BudgetExceededError):
        gw.execute("ping", {})


def test_timeout_raises_typed_error() -> None:
    reg = ToolRegistry()
    reg.register(
        _spec(
            "slow",
            "a slow tool",
            lambda a: time.sleep(0.5) or "done",
            permission=ToolPermission.ALLOW,
            timeout_s=0.05,
        )
    )
    gw = ExecuteToolGateway(reg, AuthorizationPolicy())
    with pytest.raises(ToolTimeoutError):
        gw.execute("slow", {})


def test_result_is_size_bounded() -> None:
    reg = ToolRegistry()
    reg.register(
        _spec(
            "big",
            "returns a lot",
            lambda a: "x" * 100_000,
            permission=ToolPermission.ALLOW,
            result_cap=1000,
        )
    )
    gw = ExecuteToolGateway(reg, AuthorizationPolicy())
    result = gw.execute("big", {})
    assert len(result.output) == 1000
    assert result.truncated


def test_operation_gate() -> None:
    reg = ToolRegistry()
    reg.register(
        _spec(
            "kv",
            "key-value store",
            lambda a: "ok",
            permission=ToolPermission.ALLOW,
            operations=frozenset({"read"}),
        )
    )
    gw = ExecuteToolGateway(reg, AuthorizationPolicy())
    assert gw.execute("kv", {}, operation="read").output == "ok"
    with pytest.raises(PermissionDeniedError):
        gw.execute("kv", {}, operation="delete")


# -- meta-tool round-trip ----------------------------------------------------


def test_execute_then_read_result_by_ref() -> None:
    reg = _registry()
    runtime = ToolRuntime(reg, ExecuteToolGateway(reg, AuthorizationPolicy()))
    ack = runtime.execute_tool("query_metrics", {"service": "checkout"})
    assert ack.ref.startswith("res_")
    full = runtime.read_tool_result(ack.ref)
    assert "p95=" in full.output
