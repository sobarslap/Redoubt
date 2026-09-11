"""Compactor seam — turns a stretch of conversation into a State JSON node.

Production uses an LLM compactor (structured-output prompt). The default here is
deterministic so the verification gate can be tested hermetically: it carries the
prior state's critical variables forward, absorbs explicitly-tagged updates from
the conversation, and records milestones. A *faulty* compactor that drops a
critical variable is exactly what the verifier must catch — so the seam matters.
"""

from __future__ import annotations

import re
from typing import Protocol

from aegismem.context.state import ConversationTurn, Role, StateNode

# Convention for machine-readable state hints inside turn text (the demo/tools
# emit these; the LLM compactor would infer them). e.g. "[state] db=postgres".
_STATE_HINT = re.compile(r"\[state\]\s*([\w.]+)\s*=\s*(\S+)", re.IGNORECASE)
_MILESTONE = re.compile(r"\[done\]\s*(.+)", re.IGNORECASE)


class Compactor(Protocol):
    def compact(
        self, turns: list[ConversationTurn], prior_state: StateNode | None
    ) -> StateNode: ...


class DeterministicCompactor:
    """Rule-based compactor: prior state + tagged updates from the turns."""

    def compact(self, turns: list[ConversationTurn], prior_state: StateNode | None) -> StateNode:
        state = prior_state.model_copy(deep=True) if prior_state is not None else StateNode()
        for turn in turns:
            for key, value in _STATE_HINT.findall(turn.content):
                state.critical_variables[key] = value
            for milestone in _MILESTONE.findall(turn.content):
                text = milestone.strip()
                if text not in state.completed_milestones:
                    state.completed_milestones.append(text)
            if turn.role is Role.USER and turn.content.strip():
                state.current_task = turn.content.strip()
        return state
