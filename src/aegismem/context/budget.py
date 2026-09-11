"""Context Budget Allocator — configurable percentages, not a hardcoded line.

Splits a model's usable context window across the runtime's sections (system,
state, memory, tools, conversation, reserve) by the percentages in
``config/budgets.yaml``. The allocation is fixed for a run so every subsystem
knows its ceiling up front; overspend is a typed budget error, not a surprise.
"""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_SPLIT: dict[str, float] = {
    "system": 0.15,
    "state": 0.10,
    "memory": 0.15,
    "tools": 0.15,
    "conversation": 0.20,
    "reserve": 0.25,
}


class BudgetAllocation(BaseModel):
    window: int
    sections: dict[str, int]

    def limit(self, section: str) -> int:
        return self.sections.get(section, 0)

    @property
    def usable(self) -> int:
        """Everything except the reserve headroom."""
        return sum(v for k, v in self.sections.items() if k != "reserve")


class BudgetAllocator:
    def __init__(self, split: dict[str, float] | None = None) -> None:
        split = split or DEFAULT_SPLIT
        total = sum(split.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"budget split overcommits the window: sums to {total:.3f} > 1.0")
        self._split = split

    def allocate(self, window: int) -> BudgetAllocation:
        if window <= 0:
            raise ValueError("context window must be positive")
        sections = {name: int(window * frac) for name, frac in self._split.items()}
        return BudgetAllocation(window=window, sections=sections)
