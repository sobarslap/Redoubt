"""The three meta-tools exposed to the LLM: Search_Tools, Execute_Tool,
Read_Tool_Result.

This is the entire tool surface the model sees — not the full catalog. It
discovers tools by search, executes them through the guarded gateway, and reads
bounded results back by reference. Keeping the surface to three meta-tools is what
makes progressive discovery possible; the gateway is what keeps discovery from
becoming authorization.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel

from aegismem.errors import NotFoundError
from aegismem.mcp.gateway import ExecuteToolGateway, ToolResult
from aegismem.mcp.registry import ToolRegistry


class SearchHit(BaseModel):
    name: str
    description: str


class ExecuteAck(BaseModel):
    """What Execute_Tool returns to the model: a preview + a reference to the
    full (bounded) result, so large outputs don't bloat the context."""

    ref: str
    tool: str
    preview: str
    truncated: bool
    trust: str


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, gateway: ExecuteToolGateway) -> None:
        self._registry = registry
        self._gateway = gateway
        self._results: dict[str, ToolResult] = {}

    # -- meta-tool 1: discovery (no authorization) -----------------------------

    def search_tools(self, query: str, top_k: int = 3) -> list[SearchHit]:
        return [SearchHit(**spec.compact()) for spec in self._registry.search(query, top_k=top_k)]

    def inject_schemas(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._registry.inject_schemas(query, top_k=top_k)

    # -- meta-tool 2: guarded execution ----------------------------------------

    def execute_tool(
        self, name: str, args: dict[str, Any], *, operation: str | None = None
    ) -> ExecuteAck:
        result = self._gateway.execute(name, args, operation=operation)
        ref = f"res_{uuid.uuid4().hex[:16]}"
        self._results[ref] = result
        preview = result.output[:280]
        return ExecuteAck(
            ref=ref,
            tool=result.tool,
            preview=preview,
            truncated=result.truncated or len(result.output) > len(preview),
            trust=result.trust.value,
        )

    # -- meta-tool 3: read a stored result -------------------------------------

    def read_tool_result(self, ref: str) -> ToolResult:
        try:
            return self._results[ref]
        except KeyError:
            raise NotFoundError(f"tool result {ref!r} not found", stage="mcp.read") from None
