"""Phase 4 gate: context management — budgets, watchdog, verified compaction.

Scenarios: short / long / extreme / compaction-failure / critical-state
preservation. The gate: compaction provably preserves critical state, or it does
not commit.
"""

from __future__ import annotations

import pytest

from aegismem.context.budget import DEFAULT_SPLIT, BudgetAllocator
from aegismem.context.compactor import Compactor, DeterministicCompactor
from aegismem.context.manager import ContextManager
from aegismem.context.state import ConversationTurn, Role, StateNode
from aegismem.context.tokens import HeuristicTokenCounter
from aegismem.context.verifier import verify_state
from aegismem.context.watchdog import TokenWatchdog


def _counter() -> HeuristicTokenCounter:
    return HeuristicTokenCounter()


# -- budget allocator --------------------------------------------------------


def test_budget_allocator_splits_window() -> None:
    alloc = BudgetAllocator().allocate(32_768)
    assert alloc.limit("system") == int(32_768 * 0.15)
    assert alloc.limit("reserve") == int(32_768 * 0.25)
    assert alloc.usable <= alloc.window
    assert abs(sum(DEFAULT_SPLIT.values()) - 1.0) < 1e-9


def test_budget_allocator_rejects_overcommit() -> None:
    with pytest.raises(ValueError):
        BudgetAllocator({"a": 0.7, "b": 0.7})


# -- token counting + watchdog ----------------------------------------------


def test_token_counter_is_size_based_not_turn_based() -> None:
    counter = _counter()
    assert counter.count("") == 0
    assert counter.count("a much longer sentence with several words") > counter.count("hi")


def test_watchdog_triggers_at_high_watermark() -> None:
    counter = _counter()
    wd = TokenWatchdog(counter, window=1000, high_watermark=0.75)
    assert not wd.should_compact(700)
    assert wd.should_compact(800)


# -- compaction: short / long / extreme -------------------------------------

SYSTEM = "You are the AegisMem DevOps/SRE incident agent."


def _turns(n: int) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for i in range(n):
        turns.append(ConversationTurn(role=Role.USER, content=f"user message number {i}"))
        turns.append(ConversationTurn(role=Role.ASSISTANT, content=f"assistant reply number {i}"))
    return turns


def _manager(compactor: Compactor | None = None) -> ContextManager:
    return ContextManager(compactor or DeterministicCompactor(), _counter(), keep_recent_turns=2)


def test_short_conversation_compacts_and_preserves_recent() -> None:
    turns = [
        ConversationTurn(role=Role.USER, content="incident started, checkout erroring"),
        ConversationTurn(role=Role.TOOL, content="[state] db=postgres"),
        ConversationTurn(role=Role.USER, content="what changed"),
    ]
    result = _manager().compact(
        SYSTEM, turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    assert result.committed
    assert result.state is not None
    assert result.state.critical_variables["db"] == "postgres"
    # system pinned + last 2 turns verbatim
    assert result.preserved_turns[0].role is Role.SYSTEM
    assert result.preserved_turns[-1].content == "what changed"
    assert len(result.preserved_turns) == 3


def test_long_conversation_reduces_tokens() -> None:
    turns = [ConversationTurn(role=Role.TOOL, content="[state] db=postgres"), *_turns(40)]
    result = _manager().compact(
        SYSTEM, turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    assert result.committed
    assert result.tokens_after < result.tokens_before
    assert result.reduction > 0.5


def test_extreme_conversation_still_bounded_by_recent_turns() -> None:
    turns = [ConversationTurn(role=Role.TOOL, content="[state] db=postgres"), *_turns(500)]
    result = _manager().compact(
        SYSTEM, turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    assert result.committed
    # After compaction only system + 2 recent turns + the state node remain.
    assert len(result.preserved_turns) == 3
    assert result.reduction > 0.9


# -- critical-state preservation (the gate) ---------------------------------


def test_critical_variable_survives_compaction() -> None:
    """The MySQL->Postgres story: db=postgres must survive compaction."""
    turns = [
        ConversationTurn(role=Role.TOOL, content="[state] db=mysql"),
        ConversationTurn(role=Role.TOOL, content="[state] db=postgres"),  # migration
        *_turns(30),
    ]
    result = _manager().compact(
        SYSTEM, turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    assert result.committed
    assert result.state is not None
    assert result.state.critical_variables["db"] == "postgres"


class DroppingCompactor:
    """Faulty compactor: loses the critical variable — the verifier must catch it."""

    def compact(self, turns, prior_state):
        return StateNode(current_task="continue incident", critical_variables={})


class ContradictingCompactor:
    def compact(self, turns, prior_state):
        return StateNode(current_task="continue", critical_variables={"db": "mysql"})


def test_compaction_failure_when_critical_variable_dropped() -> None:
    turns = [ConversationTurn(role=Role.TOOL, content="[state] db=postgres"), *_turns(20)]
    result = _manager(DroppingCompactor()).compact(
        SYSTEM, turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    assert not result.committed
    assert any("vanished" in r for r in result.verification.reasons)
    # previous context preserved verbatim — no lossy commit
    assert result.preserved_turns[0].role is Role.SYSTEM
    assert result.tokens_after == result.tokens_before


def test_compaction_rejected_when_state_contradicts_source_truth() -> None:
    turns = [ConversationTurn(role=Role.TOOL, content="[state] db=postgres"), *_turns(10)]
    result = _manager(ContradictingCompactor()).compact(
        SYSTEM, turns, critical_keys={"db"}, source_truth={"db": "postgres"}
    )
    assert not result.committed
    assert any("contradicts" in r for r in result.verification.reasons)


def test_verifier_direct_schema_check() -> None:
    res = verify_state(
        StateNode(current_task="", critical_variables={"db": "postgres"}),
        critical_keys={"db"},
        source_truth={"db": "postgres"},
    )
    assert not res.ok
    assert any("current_task" in r for r in res.reasons)
