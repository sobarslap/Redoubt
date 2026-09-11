"""Compaction verification — the stage that makes the Phase 4 gate real.

A compacted :class:`StateNode` is committed only if it passes:

1. **schema** — valid StateNode with a non-empty ``current_task``;
2. **preservation** — every declared critical key is still present;
3. **consistency** — each critical value matches the source-of-truth value
   (the current active fact), so compaction never silently rewrites or drops
   critical state.

Any failure returns ``ok=False`` with reasons; the caller then preserves the
previous context rather than committing a lossy summary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from aegismem.context.state import StateNode


class VerificationResult(BaseModel):
    ok: bool
    reasons: list[str] = Field(default_factory=list)


def verify_state(
    state: StateNode,
    *,
    critical_keys: set[str],
    source_truth: dict[str, str],
) -> VerificationResult:
    reasons: list[str] = []

    if not state.current_task.strip():
        reasons.append("schema: current_task is empty")

    for key in sorted(critical_keys):
        if key not in state.critical_variables:
            reasons.append(f"preservation: critical variable '{key}' vanished")
            continue
        expected = source_truth.get(key)
        actual = state.critical_variables[key]
        if expected is None:
            # A declared critical key with no source-of-truth value can't be
            # consistency-checked; fail rather than pass silently.
            reasons.append(f"consistency: no source-of-truth value for critical variable '{key}'")
        elif actual != expected:
            reasons.append(
                f"consistency: critical variable '{key}' is '{actual}', "
                f"contradicts source-of-truth '{expected}'"
            )

    return VerificationResult(ok=not reasons, reasons=reasons)
