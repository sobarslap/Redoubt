"""Phase 2 gate: memory semantics — lifecycle, provenance, conflict resolution.

Covers the required scenarios: duplicate / contradictory / updated / stale /
irrelevant fact, and asserts the ``memory_corruption = 0`` invariant (a
below-commit conflict never mutates the existing memory).
"""

from __future__ import annotations

import pytest

from aegismem.memory.conflict import (
    ConflictRelation,
    ConflictResolver,
    ConflictVerdict,
    Decision,
    HeuristicConflictClassifier,
)
from aegismem.memory.lifecycle import TransitionError, is_valid_transition
from aegismem.memory.models import MemoryCreate, MemoryStatus, MemoryType, TrustLevel
from aegismem.memory.store import SQLiteMemoryStore


@pytest.fixture
def store() -> SQLiteMemoryStore:
    s = SQLiteMemoryStore(":memory:")
    yield s
    s.close()


class StubClassifier:
    """Returns a fixed verdict — isolates the deterministic policy layer."""

    def __init__(self, relation: ConflictRelation, confidence: float) -> None:
        self._verdict = ConflictVerdict(relation=relation, confidence=confidence, rationale="stub")

    def classify(self, candidate, existing) -> ConflictVerdict:
        return self._verdict


def _active(store: SQLiteMemoryStore, content: str) -> str:
    rec = store.create(MemoryCreate(type=MemoryType.SEMANTIC, content=content))
    store.transition(rec.id, MemoryStatus.ACTIVE, reason="seed")
    return rec.id


# -- lifecycle ---------------------------------------------------------------


def test_allowed_and_forbidden_edges() -> None:
    assert is_valid_transition(MemoryStatus.CANDIDATE, MemoryStatus.ACTIVE)
    assert is_valid_transition(MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED)
    assert not is_valid_transition(MemoryStatus.DELETED, MemoryStatus.ACTIVE)
    assert not is_valid_transition(MemoryStatus.SUPERSEDED, MemoryStatus.ACTIVE)


def test_transition_writes_audit_history(store: SQLiteMemoryStore) -> None:
    mem_id = _active(store, "db is postgres")
    store.transition(mem_id, MemoryStatus.ARCHIVED, reason="incident closed", trigger="ttl")
    hist = store.history(mem_id)
    assert [h.to_status for h in hist] == [MemoryStatus.ACTIVE, MemoryStatus.ARCHIVED]
    assert hist[-1].reason == "incident closed"
    assert hist[-1].prev_version == 2


def test_illegal_transition_raises_and_does_not_mutate(store: SQLiteMemoryStore) -> None:
    rec = store.create(MemoryCreate(type=MemoryType.SEMANTIC, content="x"))
    store.transition(rec.id, MemoryStatus.ACTIVE, reason="seed")
    store.transition(rec.id, MemoryStatus.SUPERSEDED, reason="stale")
    with pytest.raises(TransitionError):
        store.transition(rec.id, MemoryStatus.ACTIVE, reason="revive")
    assert store.get(rec.id).status is MemoryStatus.SUPERSEDED  # unchanged


# -- conflict resolution -----------------------------------------------------


def test_exact_duplicate_is_rejected(store: SQLiteMemoryStore) -> None:
    existing_id = _active(store, "Primary DB is Postgres 16")
    resolver = ConflictResolver(store, HeuristicConflictClassifier())
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="primary db is postgres 16  "),
        existing=[store.get(existing_id)],
    )
    assert result.decision is Decision.REJECT_DUPLICATE
    assert result.existing_id == existing_id


def test_irrelevant_fact_is_committed_independently(store: SQLiteMemoryStore) -> None:
    existing_id = _active(store, "Primary DB is Postgres")
    resolver = ConflictResolver(store, HeuristicConflictClassifier())
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="Cache layer is Redis 7"),
        existing=[store.get(existing_id)],
    )
    assert result.decision is Decision.COMMIT_NEW
    assert result.new_memory is not None
    assert store.get(existing_id).status is MemoryStatus.ACTIVE  # untouched


def test_updated_fact_supersedes_old(store: SQLiteMemoryStore) -> None:
    old_id = _active(store, "Primary DB is MySQL 8.0")
    resolver = ConflictResolver(store, StubClassifier(ConflictRelation.UPDATES, 0.94))
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="Primary DB migrated to Postgres 16"),
        existing=[store.get(old_id)],
    )
    assert result.decision is Decision.COMMIT_SUPERSEDE
    assert result.superseded_id == old_id
    assert store.get(old_id).status is MemoryStatus.SUPERSEDED
    assert result.new_memory is not None
    assert store.get(result.new_memory.id).status is MemoryStatus.ACTIVE
    # provenance edge recorded both directions
    assert store.provenance(result.new_memory.id).supersedes == [old_id]
    assert store.provenance(old_id).superseded_by == [result.new_memory.id]


def test_supersede_of_candidate_routes_to_review_not_corruption(store: SQLiteMemoryStore) -> None:
    # A CANDIDATE record is 'live' for conflict candidacy but cannot legally reach
    # SUPERSEDED. A high-confidence UPDATES verdict against it must NOT attempt the
    # illegal transition (which would leave a half-applied ACTIVE new record and two
    # contradictory live memories) — it parks for review instead. Upholds
    # memory_corruption = 0.
    cand = store.create(MemoryCreate(type=MemoryType.SEMANTIC, content="Primary DB is MySQL 8.0"))
    assert store.get(cand.id).status is MemoryStatus.CANDIDATE
    resolver = ConflictResolver(store, StubClassifier(ConflictRelation.UPDATES, 0.94))
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="Primary DB migrated to Postgres 16"),
        existing=[store.get(cand.id)],
    )
    assert result.decision is Decision.HUMAN_REVIEW  # not COMMIT_SUPERSEDE
    assert store.get(cand.id).status is MemoryStatus.CANDIDATE  # untouched
    # No new ACTIVE memory was created behind a failed supersede.
    assert store.list(status=MemoryStatus.ACTIVE) == []


def test_high_confidence_contradiction_supersedes(store: SQLiteMemoryStore) -> None:
    old_id = _active(store, "Service runs in us-east-1")
    resolver = ConflictResolver(store, StubClassifier(ConflictRelation.CONTRADICTS, 0.9))
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="Service runs in eu-west-1"),
        existing=[store.get(old_id)],
    )
    assert result.decision is Decision.COMMIT_SUPERSEDE
    assert store.get(old_id).status is MemoryStatus.SUPERSEDED


def test_mid_confidence_conflict_goes_to_review_without_mutation(store: SQLiteMemoryStore) -> None:
    old_id = _active(store, "Deploy cadence is weekly")
    resolver = ConflictResolver(store, StubClassifier(ConflictRelation.CONTRADICTS, 0.7))
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="Deploy cadence is daily"),
        existing=[store.get(old_id)],
    )
    assert result.decision is Decision.HUMAN_REVIEW
    assert result.review_id is not None
    # Invariant: memory corruption = 0 — existing memory is not touched.
    assert store.get(old_id).status is MemoryStatus.ACTIVE
    queue = store.review_queue()
    assert len(queue) == 1
    assert queue[0]["existing_id"] == old_id


def test_low_confidence_conflict_commits_new(store: SQLiteMemoryStore) -> None:
    old_id = _active(store, "Runbook step 3 is restart")
    resolver = ConflictResolver(store, StubClassifier(ConflictRelation.CONTRADICTS, 0.4))
    result = resolver.ingest(
        MemoryCreate(type=MemoryType.SEMANTIC, content="Runbook step 3 is failover"),
        existing=[store.get(old_id)],
    )
    assert result.decision is Decision.COMMIT_NEW
    assert store.get(old_id).status is MemoryStatus.ACTIVE


def test_derived_from_edges_are_recorded(store: SQLiteMemoryStore) -> None:
    parent_id = _active(store, "raw metric snapshot")
    resolver = ConflictResolver(store, HeuristicConflictClassifier())
    result = resolver.ingest(
        MemoryCreate(
            type=MemoryType.SEMANTIC,
            content="p95 latency exceeded SLO",
            source="tool:metrics_api",
            trust=TrustLevel.TOOL_RESULT,
            derived_from=[parent_id],
        ),
        existing=[],
    )
    assert result.new_memory is not None
    assert store.provenance(result.new_memory.id).derived_from == [parent_id]
