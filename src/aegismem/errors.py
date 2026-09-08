"""Typed error envelope and the exception that carries it.

The runtime contract mandates a single typed error shape across every stage:
``{ code, category, message, retryable, stage }`` — never a raw stack trace.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ErrorCategory(StrEnum):
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    GUARDRAIL = "guardrail"
    BUDGET = "budget"
    TOOL = "tool"
    LLM = "llm"
    STORAGE = "storage"
    INTERNAL = "internal"


class ErrorEnvelope(BaseModel):
    """Serializable error returned to clients and threaded through traces."""

    code: str
    category: ErrorCategory
    message: str
    retryable: bool = False
    stage: str | None = None


class AegisError(Exception):
    """Base exception carrying an ``ErrorEnvelope``."""

    def __init__(
        self,
        code: str,
        category: ErrorCategory,
        message: str,
        *,
        retryable: bool = False,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.envelope = ErrorEnvelope(
            code=code,
            category=category,
            message=message,
            retryable=retryable,
            stage=stage,
        )


class NotFoundError(AegisError):
    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__("not_found", ErrorCategory.NOT_FOUND, message, stage=stage)


class StorageError(AegisError):
    def __init__(self, message: str, *, retryable: bool = True, stage: str | None = None) -> None:
        super().__init__(
            "storage_error", ErrorCategory.STORAGE, message, retryable=retryable, stage=stage
        )


class PermissionDeniedError(AegisError):
    """Raised by the Execute_Tool gateway when authorization is refused.

    Uphold of ``unauthorized_tool_exec = 0`` — the tool handler never runs.
    """

    def __init__(self, message: str) -> None:
        super().__init__("unauthorized_tool", ErrorCategory.TOOL, message, stage="mcp.execute")


class CancelledError(AegisError):
    """Raised at a stage boundary when a run has been cooperatively cancelled."""

    def __init__(self, message: str = "run cancelled", *, stage: str | None = None) -> None:
        super().__init__("cancelled", ErrorCategory.INTERNAL, message, stage=stage)


class GuardrailError(AegisError):
    """Raised when the security boundary refuses input, a memory write, a tool
    result, or an output.

    Fail-closed: when a guardrail cannot decide (internal error, unavailable
    classifier), it raises this rather than allowing the content through.
    """

    def __init__(self, message: str, *, stage: str = "guardrails") -> None:
        super().__init__("guardrail_blocked", ErrorCategory.GUARDRAIL, message, stage=stage)


class ArgumentValidationError(AegisError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "invalid_arguments", ErrorCategory.VALIDATION, message, stage="mcp.execute"
        )


class BudgetExceededError(AegisError):
    def __init__(self, message: str) -> None:
        super().__init__("budget_exceeded", ErrorCategory.BUDGET, message, stage="mcp.execute")


class ToolTimeoutError(AegisError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "tool_timeout", ErrorCategory.TOOL, message, retryable=True, stage="mcp.execute"
        )


class ToolExecutionError(AegisError):
    def __init__(self, message: str) -> None:
        super().__init__("tool_failed", ErrorCategory.TOOL, message, stage="mcp.execute")
