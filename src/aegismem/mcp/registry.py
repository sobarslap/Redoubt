"""Tool registry + progressive schema injection.

Full tool schemas live in a local offline index. Only the **top-K matching
schemas** are injected into the model context on demand, collapsing 60k+ chars of
catalog to <2k. Search is BM25 over name + description (reusing the retrieval
index). A :class:`ToolSpec` also carries its *security* metadata — permission,
timeouts, rate limit, result-size cap, result trust level — kept next to the
schema so the gateway has everything it needs to guard execution.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from aegismem.errors import NotFoundError
from aegismem.memory.models import TrustLevel
from aegismem.retrieval.bm25 import BM25Index


class ToolPermission(StrEnum):
    ALLOW = "allow"  # authorized when on the run's allow-list / default policy
    REQUIRE_APPROVAL = "require_approval"  # never auto-authorized
    DENY = "deny"  # always refused


@dataclass(slots=True)
class ToolSpec:
    """A tool's schema plus the security metadata the gateway enforces."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON-schema-shaped
    handler: Callable[[dict[str, Any]], Any]
    args_model: type[BaseModel] | None = None
    permission: ToolPermission = ToolPermission.REQUIRE_APPROVAL
    timeout_s: float = 5.0
    max_executions_per_run: int = 3
    result_size_cap: int = 8000
    result_trust: TrustLevel = TrustLevel.TOOL_RESULT
    operations: frozenset[str] = field(default_factory=frozenset)

    def compact(self) -> dict[str, str]:
        """The tiny discovery view (name + description only)."""
        return {"name": self.name, "description": self.description}

    def full_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def schema_chars(self) -> int:
        return len(json.dumps(self.full_schema()))


class ToolRegistry:
    """Holds tool specs and a BM25 search index over name + description."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._index = BM25Index()

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec
        self._index.add(spec.name, f"{spec.name} {spec.description}")

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise NotFoundError(f"tool {name!r} not registered", stage="mcp.registry") from None

    def search(self, query: str, top_k: int = 3) -> list[ToolSpec]:
        """Discovery — returns matching specs regardless of authorization."""
        ranked = self._index.search(query, top_k=top_k)
        return [self._specs[name] for name, _ in ranked]

    def inject_schemas(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Full schemas for the top-K matches only (the injection payload)."""
        return [spec.full_schema() for spec in self.search(query, top_k=top_k)]

    def catalog_schema_chars(self) -> int:
        """Total chars if every schema were injected — the 'before' number."""
        return sum(spec.schema_chars() for spec in self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
