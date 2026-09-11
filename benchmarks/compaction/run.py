"""Compaction-effectiveness benchmark — token reduction with critical state kept.

Grows a synthetic incident conversation and measures the verified compactor's
token reduction while asserting the critical variable survives. Naive baseline =
no compaction (keep the whole transcript); AegisMem = compact to a verified State
node + recent turns.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegismem.context.compactor import DeterministicCompactor
from aegismem.context.manager import ContextManager
from aegismem.context.state import ConversationTurn, Role
from aegismem.context.tokens import HeuristicTokenCounter


@dataclass(frozen=True)
class CompactionPoint:
    turns: int
    tokens_before: int
    tokens_after: int
    reduction: float
    critical_preserved: bool
    committed: bool


def _conversation(n_turns: int) -> list[ConversationTurn]:
    turns = [ConversationTurn(role=Role.USER, content="diagnose the checkout outage")]
    turns.append(ConversationTurn(role=Role.ASSISTANT, content="confirmed [state] db=postgres"))
    turns += [
        ConversationTurn(role=Role.ASSISTANT, content=f"log window {i}: " + "analysis noise " * 30)
        for i in range(n_turns)
    ]
    return turns


def compaction_point(n_turns: int) -> CompactionPoint:
    mgr = ContextManager(DeterministicCompactor(), HeuristicTokenCounter(), keep_recent_turns=2)
    result = mgr.compact(
        "You are an SRE assistant.",
        _conversation(n_turns),
        critical_keys={"db"},
        source_truth={"db": "postgres"},
    )
    preserved = result.state is not None and result.state.critical_variables.get("db") == "postgres"
    return CompactionPoint(
        turns=n_turns,
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        reduction=round(result.reduction, 6),
        critical_preserved=preserved,
        committed=result.committed,
    )


def main(turn_counts: list[int] | None = None) -> None:
    turn_counts = turn_counts or [20, 50, 100, 200]
    print("compaction - token reduction (critical state must survive):")
    print(f"{'turns':>8} {'before':>10} {'after':>10} {'reduction':>10} {'critical':>10}")
    for n in turn_counts:
        p = compaction_point(n)
        crit = "kept" if p.critical_preserved else "LOST"
        print(
            f"{p.turns:>8} {p.tokens_before:>10} {p.tokens_after:>10} "
            f"{p.reduction:>9.2%} {crit:>10}"
        )


if __name__ == "__main__":
    main()
