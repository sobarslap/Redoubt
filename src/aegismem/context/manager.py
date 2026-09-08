"""Context manager — orchestrates the verified compaction run.

    Conversation -> Compactor -> State JSON -> schema validation
                 -> consistency check -> commit (or preserve previous)

The system prompt stays pinned at the top and the last ``keep_recent_turns`` are
kept verbatim; everything before them collapses into the verified State node.
Compaction commits only if verification passes — otherwise the previous context
is preserved and the failure is reported (the "compaction -> preserve previous,
alert" policy). **Gate: compaction provably preserves critical state.**
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from aegismem.context.compactor import Compactor
from aegismem.context.state import ConversationTurn, Role, StateNode
from aegismem.context.tokens import TokenCounter
from aegismem.context.verifier import VerificationResult, verify_state


class CompactionResult(BaseModel):
    committed: bool
    state: StateNode | None = None
    preserved_turns: list[ConversationTurn] = Field(default_factory=list)
    verification: VerificationResult
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def reduction(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before


class ContextManager:
    def __init__(
        self,
        compactor: Compactor,
        counter: TokenCounter,
        *,
        keep_recent_turns: int = 2,
    ) -> None:
        self._compactor = compactor
        self._counter = counter
        self._keep = keep_recent_turns

    def _tokens(self, turns: list[ConversationTurn], state: StateNode | None) -> int:
        total = sum(self._counter.count(t.content) for t in turns)
        if state is not None:
            total += self._counter.count(state.model_dump_json())
        return total

    def compact(
        self,
        system_prompt: str,
        turns: list[ConversationTurn],
        *,
        prior_state: StateNode | None = None,
        critical_keys: set[str],
        source_truth: dict[str, str],
    ) -> CompactionResult:
        system_turn = ConversationTurn(role=Role.SYSTEM, content=system_prompt)
        # The State node summarizes the *whole* conversation; the last N turns are
        # ADDITIONALLY kept verbatim for fidelity. So compaction reads every turn,
        # while preservation keeps system + recent tail verbatim.
        recent = turns[-self._keep :] if self._keep > 0 else []

        tokens_before = self._counter.count(system_prompt) + self._tokens(turns, prior_state)

        new_state = self._compactor.compact(turns, prior_state)
        verification = verify_state(
            new_state, critical_keys=critical_keys, source_truth=source_truth
        )

        if not verification.ok:
            # Preserve previous context; do not commit a lossy/invalid summary.
            preserved = [system_turn, *turns]
            return CompactionResult(
                committed=False,
                state=prior_state,
                preserved_turns=preserved,
                verification=verification,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        preserved = [system_turn, *recent]
        tokens_after = self._counter.count(system_prompt) + self._tokens(recent, new_state)
        return CompactionResult(
            committed=True,
            state=new_state,
            preserved_turns=preserved,
            verification=verification,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
