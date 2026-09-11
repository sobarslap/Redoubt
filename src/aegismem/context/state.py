"""Structured conversation state — a State JSON node, never lossy prose.

Compaction replaces a stretch of conversation with a typed :class:`StateNode`
(current task, completed milestones, and — critically — the ``critical_variables``
that must survive). Because it is structured, the verification stage can check it
by schema and by value, which prose can never guarantee.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationTurn(BaseModel):
    role: Role
    content: str


class StateNode(BaseModel):
    """The compacted representation of prior conversation."""

    current_task: str = ""
    completed_milestones: list[str] = Field(default_factory=list)
    critical_variables: dict[str, str] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
