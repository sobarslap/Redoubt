"""Execute_Tool gateway — discovery is not authorization.

The guarded pipeline, in order::

    allowed? (allow/deny list + per-tool permission)
      -> args valid? (Pydantic)
      -> operation permitted?
      -> within budget? (per-tool count, per-run count, wall time)
      -> EXECUTE (sandboxed thread, timeout)
      -> validate + bound result

The LLM never makes the authorization decision — this gateway does. The handler
is reached only after every check passes, which is what upholds
**unauthorized_tool_exec = 0**.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from aegismem.errors import (
    ArgumentValidationError,
    BudgetExceededError,
    PermissionDeniedError,
    ToolExecutionError,
    ToolTimeoutError,
)
from aegismem.mcp.registry import ToolPermission, ToolRegistry, ToolSpec
from aegismem.memory.models import TrustLevel


@dataclass(slots=True)
class AuthorizationPolicy:
    """Explicit allow/deny lists layered over each tool's default permission."""

    allow: frozenset[str] | None = None  # None => defer to per-tool permission
    deny: frozenset[str] = field(default_factory=frozenset)

    def is_authorized(self, spec: ToolSpec) -> tuple[bool, str]:
        if spec.name in self.deny:
            return False, "tool is on the deny-list"
        if spec.permission is ToolPermission.DENY:
            return False, "tool permission is DENY"
        if self.allow is not None:
            if spec.name in self.allow:
                return True, "on allow-list"
            return False, "not on allow-list"
        # No explicit allow-list: only default-ALLOW tools are authorized.
        if spec.permission is ToolPermission.ALLOW:
            return True, "default permission ALLOW"
        return False, "requires approval (not on allow-list)"


@dataclass(slots=True)
class RunBudget:
    max_tool_calls: int = 12
    max_wall_seconds: float = 30.0
    _calls: int = 0
    _per_tool: dict[str, int] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    def check_and_reserve(self, spec: ToolSpec) -> None:
        if self._calls >= self.max_tool_calls:
            raise BudgetExceededError(f"run tool-call budget exhausted ({self.max_tool_calls})")
        if time.perf_counter() - self._started > self.max_wall_seconds:
            raise BudgetExceededError("run wall-time budget exhausted")
        used = self._per_tool.get(spec.name, 0)
        if used >= spec.max_executions_per_run:
            raise BudgetExceededError(
                f"per-tool budget exhausted for {spec.name!r} ({spec.max_executions_per_run})"
            )
        self._calls += 1
        self._per_tool[spec.name] = used + 1


class ToolResult(BaseModel):
    tool: str
    output: str
    trust: TrustLevel
    truncated: bool = False
    latency_ms: float = 0.0
    authorized_by: str = ""
    operation: str | None = None


class ExecuteToolGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: AuthorizationPolicy,
        budget: RunBudget | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._budget = budget or RunBudget()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.executions = 0  # count of handler invocations that actually ran

    def execute(
        self, name: str, args: dict[str, Any], *, operation: str | None = None
    ) -> ToolResult:
        spec = self._registry.get(name)  # NotFoundError if unknown

        # 1. Authorization — the gateway decides, not the caller.
        authorized, reason = self._policy.is_authorized(spec)
        if not authorized:
            raise PermissionDeniedError(f"execution of {name!r} refused: {reason}")

        # 2. Argument validation.
        validated = self._validate_args(spec, args)

        # 3. Operation permitted?
        if operation is not None and spec.operations and operation not in spec.operations:
            raise PermissionDeniedError(f"operation {operation!r} not permitted for tool {name!r}")

        # 4. Budget (count + wall time). Reserves the slot before running.
        self._budget.check_and_reserve(spec)

        # 5. Sandboxed execution with timeout.
        start = time.perf_counter()
        try:
            future = self._executor.submit(spec.handler, validated)
            raw = future.result(timeout=spec.timeout_s)
        except FutureTimeout as exc:
            raise ToolTimeoutError(f"tool {name!r} exceeded timeout {spec.timeout_s}s") from exc
        except Exception as exc:  # handler blew up
            raise ToolExecutionError(f"tool {name!r} failed: {exc}") from exc
        self.executions += 1
        latency_ms = (time.perf_counter() - start) * 1000

        # 6. Validate + bound the result.
        output = raw if isinstance(raw, str) else str(raw)
        truncated = len(output) > spec.result_size_cap
        if truncated:
            output = output[: spec.result_size_cap]

        return ToolResult(
            tool=name,
            output=output,
            trust=spec.result_trust,
            truncated=truncated,
            latency_ms=round(latency_ms, 4),
            authorized_by=reason,
            operation=operation,
        )

    @staticmethod
    def _validate_args(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        if spec.args_model is None:
            return args
        try:
            model = spec.args_model.model_validate(args)
        except ValidationError as exc:
            raise ArgumentValidationError(
                f"invalid arguments for {spec.name!r}: {exc.error_count()} error(s)"
            ) from exc
        return model.model_dump()
