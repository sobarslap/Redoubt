"""SecurityBoundary — the single façade the runtime contract calls.

Wires the layered defenses into the two enforcement points named in the runtime
contract:

    Request → Validation → **Guardrails (input)** → Budget → Plan → Execute
            → **Output validation (leakage filter)** → Response

Layers, in order of the plan's §6:

    privilege separation (trust labels + structural assembly)
      → input sanitization (resource limits, injection scan, PII)
      → prompt hardening / scope (immutable rules, fenced untrusted data)
      → output validation / authorization (leakage filter)

Constructed from ``policies.yaml`` so thresholds are configuration, not literals.
Fail-closed by default: oversized input is refused rather than truncated-and-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aegismem.config.settings import Settings, get_settings
from aegismem.errors import GuardrailError
from aegismem.guardrails import pii
from aegismem.guardrails.injection import InjectionVerdict, PromptInjectionScanner
from aegismem.guardrails.output import OutputVerdict, filter_output
from aegismem.guardrails.sanitize import (
    MemoryWriteSanitizer,
    ToolResultReport,
    ToolResultValidator,
    WriteReport,
)
from aegismem.guardrails.trust import Segment, assemble_context
from aegismem.mcp.gateway import ToolResult
from aegismem.memory.models import MemoryCreate, TrustLevel


@dataclass(slots=True, frozen=True)
class InputDecision:
    """Result of inspecting untrusted user/RAG input at the boundary."""

    allowed: bool
    injection: InjectionVerdict
    pii_kinds: list[str]
    reason: str = ""


class SecurityBoundary:
    def __init__(self, settings: Settings | None = None) -> None:
        cfg: dict[str, Any] = (settings or get_settings()).policies().get("security", {})
        self._max_input = int(cfg.get("max_input_chars", 200000))
        self._fail_closed = bool(cfg.get("fail_closed", True))
        block = float(cfg.get("injection_block_threshold", 0.5))
        ceiling = TrustLevel(cfg.get("untrusted_write_max_trust", "user"))

        self.scanner = PromptInjectionScanner(block_threshold=block)
        self.memory_writes = MemoryWriteSanitizer(
            scanner=self.scanner,
            max_content_chars=int(cfg.get("max_memory_content_chars", 20000)),
            untrusted_write_max_trust=ceiling,
        )
        self.tool_results = ToolResultValidator(
            scanner=self.scanner,
            max_result_chars=int(cfg.get("max_tool_result_chars", 32000)),
        )

    # -- input side ----------------------------------------------------------

    def inspect_input(self, text: str, *, trust: TrustLevel = TrustLevel.USER) -> InputDecision:
        """Resource limit + injection scan + PII scan on inbound content.

        For USER-trust input, an injection payload is *flagged* but not refused
        (a user may legitimately paste a suspicious log to ask about it). For
        untrusted data (RAG/tool), the decision is advisory — the structural
        fence in ``assemble_context`` is what actually neutralizes it.
        """
        if len(text) > self._max_input:
            raise GuardrailError(
                f"input exceeds max_input_chars ({len(text)} > {self._max_input})",
                stage="guardrails.input",
            )
        verdict = self.scanner.scan(text)
        pii_kinds = sorted({f.kind for f in pii.scan(text)})
        return InputDecision(
            allowed=True,
            injection=verdict,
            pii_kinds=pii_kinds,
            reason="flagged" if verdict.blocked else "",
        )

    def assemble(
        self, *, system: str, developer: str = "", segments: list[Segment], nonce: str | None = None
    ) -> str:
        return assemble_context(system=system, developer=developer, segments=segments, nonce=nonce)

    # -- ingest side ---------------------------------------------------------

    def sanitize_memory_write(self, data: MemoryCreate) -> WriteReport:
        return self.memory_writes.sanitize(data)

    def validate_tool_result(self, result: ToolResult) -> ToolResultReport:
        return self.tool_results.validate(result)

    # -- output side ---------------------------------------------------------

    def filter_output(
        self, text: str, *, secrets: frozenset[str] = frozenset(), redact_pii: bool = True
    ) -> OutputVerdict:
        return filter_output(text, secrets=secrets, redact_pii=redact_pii)
