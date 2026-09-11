"""Deterministic conflict resolution — the LLM assists, it never decides alone.

Pipeline::

    new fact -> exact-duplicate check -> classify vs candidates
             -> confidence threshold -> COMMIT / REJECT / HUMAN_REVIEW

A classifier (LLM in production, an injectable stub or heuristic here) returns a
typed verdict ``{relation, confidence, rationale}``. A *deterministic policy
layer* — not the classifier — maps (verdict, confidence) onto a state-machine
action. Below the review threshold nothing mutates: the candidate is committed as
independent, and a mid-confidence conflict is parked in the human-review queue
rather than silently rewriting memory. **Invariant: memory corruption = 0.**
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from aegismem.memory.lifecycle import LIVE_STATUSES
from aegismem.memory.models import MemoryCreate, MemoryRecord, MemoryStatus


class ConflictRelation(StrEnum):
    DUPLICATE = "duplicate"
    CONTRADICTS = "contradicts"
    UPDATES = "updates"
    UNRELATED = "unrelated"


class ConflictVerdict(BaseModel):
    relation: ConflictRelation
    confidence: float
    rationale: str = ""


class Decision(StrEnum):
    COMMIT_NEW = "commit_new"  # no conflict — added as independent, ACTIVE
    COMMIT_SUPERSEDE = "commit_supersede"  # new ACTIVE, old -> SUPERSEDED
    REJECT_DUPLICATE = "reject_duplicate"  # existing kept, new dropped
    HUMAN_REVIEW = "human_review"  # parked, nothing mutated


class ResolutionResult(BaseModel):
    decision: Decision
    verdict: ConflictVerdict
    new_memory: MemoryRecord | None = None
    superseded_id: str | None = None
    existing_id: str | None = None
    review_id: str | None = None


class ConflictClassifier(Protocol):
    """Assesses the relation between a candidate and one existing memory."""

    def classify(self, candidate: MemoryCreate, existing: MemoryRecord) -> ConflictVerdict: ...


def normalize(text: str) -> str:
    """Case/whitespace-insensitive canonical form for exact-duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


class HeuristicConflictClassifier:
    """No-LLM default: detects (near-)duplicates only, never fabricates a conflict.

    Real contradiction/update detection needs semantic understanding (the LLM
    classifier, wired in later). Being conservative here keeps the policy layer
    honest: it returns ``UNRELATED`` unless token overlap is high enough to call a
    duplicate, so the resolver adds rather than mutates when it cannot be sure.
    """

    def __init__(self, duplicate_jaccard: float = 0.9) -> None:
        self._threshold = duplicate_jaccard

    def classify(self, candidate: MemoryCreate, existing: MemoryRecord) -> ConflictVerdict:
        a = set(normalize(candidate.content).split())
        b = set(normalize(existing.content).split())
        if not a or not b:
            return ConflictVerdict(relation=ConflictRelation.UNRELATED, confidence=1.0)
        jaccard = len(a & b) / len(a | b)
        if jaccard >= self._threshold:
            return ConflictVerdict(
                relation=ConflictRelation.DUPLICATE,
                confidence=jaccard,
                rationale=f"token jaccard {jaccard:.2f}",
            )
        return ConflictVerdict(relation=ConflictRelation.UNRELATED, confidence=1.0 - jaccard)


# Structural typing avoids a hard import cycle with the store module.
class _StoreLike(Protocol):
    def create(self, data: MemoryCreate) -> MemoryRecord: ...
    def transition(
        self, memory_id: str, to_status: MemoryStatus, *, reason: str, source: str, trigger: str
    ) -> MemoryRecord: ...
    def add_edge(self, src_id: str, dst_id: str, kind: str) -> None: ...
    def enqueue_review(
        self, candidate: MemoryCreate, existing_id: str, verdict: ConflictVerdict
    ) -> str: ...


class ConflictResolver:
    """Maps classifier verdicts onto state-machine actions via fixed thresholds."""

    def __init__(
        self,
        store: _StoreLike,
        classifier: ConflictClassifier,
        *,
        commit_threshold: float = 0.85,
        review_threshold: float = 0.60,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._commit = commit_threshold
        self._review = review_threshold

    def ingest(self, candidate: MemoryCreate, existing: list[MemoryRecord]) -> ResolutionResult:
        # Reuse the canonical live-status set so this never drifts from lifecycle.py
        # (which also counts VALIDATING as live).
        live = [e for e in existing if e.status in LIVE_STATUSES]

        # 1. Exact-duplicate short-circuit (deterministic, no classifier).
        norm = normalize(candidate.content)
        for e in live:
            if normalize(e.content) == norm:
                return ResolutionResult(
                    decision=Decision.REJECT_DUPLICATE,
                    verdict=ConflictVerdict(
                        relation=ConflictRelation.DUPLICATE,
                        confidence=1.0,
                        rationale="exact content match",
                    ),
                    existing_id=e.id,
                )

        # 2. Classify against each live candidate; keep the strongest conflict.
        best: tuple[MemoryRecord, ConflictVerdict] | None = None
        for e in live:
            verdict = self._classifier.classify(candidate, e)
            if verdict.relation is ConflictRelation.UNRELATED:
                continue
            if best is None or verdict.confidence > best[1].confidence:
                best = (e, verdict)

        if best is None:
            return self._commit_new(
                candidate,
                ConflictVerdict(
                    relation=ConflictRelation.UNRELATED,
                    confidence=1.0,
                    rationale="no live conflict",
                ),
            )

        existing_rec, verdict = best

        # 3. Deterministic policy layer.
        if verdict.confidence >= self._commit:
            if verdict.relation in {ConflictRelation.UPDATES, ConflictRelation.CONTRADICTS}:
                # Only an ACTIVE record can legally reach SUPERSEDED. A not-yet-active
                # candidate (CANDIDATE/VALIDATING) is in `live` too; superseding it
                # would raise mid-mutation and leave a half-applied ACTIVE new record
                # (breaking memory_corruption = 0). Park those for human review instead.
                if existing_rec.status is MemoryStatus.ACTIVE:
                    return self._commit_supersede(candidate, existing_rec, verdict)
                review_id = self._store.enqueue_review(candidate, existing_rec.id, verdict)
                return ResolutionResult(
                    decision=Decision.HUMAN_REVIEW,
                    verdict=verdict,
                    existing_id=existing_rec.id,
                    review_id=review_id,
                )
            if verdict.relation is ConflictRelation.DUPLICATE:
                return ResolutionResult(
                    decision=Decision.REJECT_DUPLICATE,
                    verdict=verdict,
                    existing_id=existing_rec.id,
                )

        if verdict.confidence >= self._review:
            review_id = self._store.enqueue_review(candidate, existing_rec.id, verdict)
            return ResolutionResult(
                decision=Decision.HUMAN_REVIEW,
                verdict=verdict,
                existing_id=existing_rec.id,
                review_id=review_id,
            )

        # Below the review threshold: not confident enough to touch memory. Add new.
        return self._commit_new(candidate, verdict)

    # -- actions ---------------------------------------------------------------

    def _commit_new(self, candidate: MemoryCreate, verdict: ConflictVerdict) -> ResolutionResult:
        # Always create as CANDIDATE, then promote — a caller-supplied ACTIVE status
        # would otherwise make the transition below an illegal ACTIVE -> ACTIVE edge.
        rec = self._store.create(candidate.model_copy(update={"status": MemoryStatus.CANDIDATE}))
        rec = self._store.transition(
            rec.id,
            MemoryStatus.ACTIVE,
            reason="committed as independent memory",
            source="conflict_resolver",
            trigger="ingest",
        )
        for parent in candidate.derived_from:
            self._store.add_edge(rec.id, parent, "derived_from")
        return ResolutionResult(decision=Decision.COMMIT_NEW, verdict=verdict, new_memory=rec)

    def _commit_supersede(
        self, candidate: MemoryCreate, existing: MemoryRecord, verdict: ConflictVerdict
    ) -> ResolutionResult:
        # Create the replacement as a (non-active) candidate first, then retire the
        # existing record, and only then activate the new one. Ordered this way, a
        # failure at any step never leaves two ACTIVE contradictory memories: the
        # worst case is a stray CANDIDATE, which is not active truth.
        new = self._store.create(
            candidate.model_copy(
                update={"supersedes": existing.id, "status": MemoryStatus.CANDIDATE}
            )
        )
        self._store.transition(
            existing.id,
            MemoryStatus.SUPERSEDED,
            reason=f"superseded by {new.id} ({verdict.relation}, conf {verdict.confidence:.2f})",
            source="conflict_resolver",
            trigger="ingest",
        )
        new = self._store.transition(
            new.id,
            MemoryStatus.ACTIVE,
            reason=f"supersedes {existing.id} ({verdict.relation})",
            source="conflict_resolver",
            trigger="ingest",
        )
        self._store.add_edge(new.id, existing.id, "supersedes")
        return ResolutionResult(
            decision=Decision.COMMIT_SUPERSEDE,
            verdict=verdict,
            new_memory=new,
            superseded_id=existing.id,
            existing_id=existing.id,
        )
