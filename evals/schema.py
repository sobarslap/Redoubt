"""Golden-dataset schema and the report the CI gate reads.

A scenario is a compact, self-describing assertion; the category names which
evaluator interprets its ``params`` and ``expect``. Evaluators own the subsystem
fixtures so scenarios stay small and legible.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    COMPACTION = "compaction"
    TOOLS = "tools"
    SECURITY = "security"
    LONG_CONTEXT = "long_context"
    FAILURE_RECOVERY = "failure_recovery"
    COST = "cost"
    LATENCY = "latency"


class Scenario(BaseModel):
    id: str
    category: Category
    description: str = ""
    params: dict[str, object] = Field(default_factory=dict)
    expect: dict[str, object] = Field(default_factory=dict)


class EvalResult(BaseModel):
    scenario_id: str
    category: Category
    passed: bool
    score: float = 1.0  # 1.0 pass / 0.0 fail for boolean checks
    detail: str = ""
    # Cross-cutting invariants this scenario touched (0 = clean).
    memory_corruption: int = 0
    unauthorized_tool_exec: int = 0
    attack_success: int = 0


class CategoryReport(BaseModel):
    category: Category
    total: int
    passed: int
    gate: float  # required pass-rate to not block CI

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    @property
    def ok(self) -> bool:
        return self.pass_rate >= self.gate


class Report(BaseModel):
    total: int
    passed: int
    categories: list[CategoryReport]
    memory_corruption: int
    unauthorized_tool_exec: int
    attack_success: int
    failures: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            all(c.ok for c in self.categories)
            and self.memory_corruption == 0
            and self.unauthorized_tool_exec == 0
            and self.attack_success == 0
        )
