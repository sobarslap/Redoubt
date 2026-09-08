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
