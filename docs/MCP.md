# MCP — Progressive Tool Discovery

The MCP subsystem (`src/aegismem/mcp/`) gives the model tools without paying the
prompt cost of every schema, and — critically — keeps **discovery separate from
authorization**.

## Three meta-tools

The model sees only three tools (`runtime.py`, `ToolRuntime`), never the full
catalog:

- **`Search_Tools`** — BM25 search over the tool registry; returns the top-K
  matching names + descriptions.
- **`Execute_Tool`** — the guarded gateway (below); the only way a tool runs.
- **`Read_Tool_Result`** — fetch a bounded result by reference, so large outputs
  don't bloat the context.

Full schemas live in a local registry (`registry.py`); only the top-K matching
schemas are injected on demand. This is the "60k+ chars → <2k" reduction measured
in `docs/PERFORMANCE.md` — injected schema size stays flat as the catalog grows.

## Discovery ≠ authorization (the security spine)

Surfacing a tool in a search **never** makes it callable. `Execute_Tool`
(`gateway.py`, `ExecuteToolGateway`) is a guarded gateway, and the LLM never makes
the authorization decision — the gateway does, in this fixed order:

```
Execute_Tool → authorized? (allow/deny list + per-tool permission)
            → args valid? (Pydantic)
            → operation permitted?
            → within budget? (per-tool count, per-run count, wall time)
            → EXECUTE (sandboxed thread, timeout)
            → validate + size-bound result
```

Any failed check raises a typed error and the handler **never runs**. Tool
permissions are `ALLOW` (authorized by default policy), `REQUIRE_APPROVAL` (never
auto-authorized — must be on the run's allow-list), or `DENY` (always refused); a
deny-list overrides everything. This upholds **unauthorized_tool_exec = 0**.

## Results are untrusted data

Tool output returns at `TOOL_RESULT` trust — untrusted. The guardrails
`ToolResultValidator` scans it for embedded instructions, and the trust boundary
fences it as data, so a poisoned log line telling the agent to "restart prod" is
refused by the gateway regardless of whether the model was fooled (proven by the
end-to-end red-team, `docs/SECURITY.md`).

## The reason↔tool loop

The execution agent (`execution/agent.py`) runs a bounded loop: the model proposes
`ToolCall`s, each executes only through the gateway, and results (or refusals) are
fed back as data until a final answer or the tool-step budget is hit. A `CostGuard`
enforces the per-run token/spend ceiling across the loop.

## Tests

`tests/unit/test_mcp.py` (discovery, the discovery/authorization split, every
gateway guard — allow/deny/permission/operation/budget/timeout/result-cap) and
the LLM-driven guarded loop in `tests/integration/test_llm_providers.py`.
