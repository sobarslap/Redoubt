"""Memory-write and tool-result sanitization — the untrusted-ingest choke point.

Every write into memory and every tool result passes through here before it can
influence the run. This is what upholds **memory_corruption = 0**: untrusted
content can never be committed as a high-trust fact, and injection-bearing
untrusted writes are downgraded to ``CANDIDATE``/review rather than landing as
``ACTIVE`` truth.

Two invariants enforced deterministically (no LLM in this path):

* **Trust ceiling** — a write whose source is untrusted (MEMORY / TOOL_RESULT /
  RAG_CONTENT) can never claim SYSTEM or DEVELOPER trust. Attempted escalation is
  clamped to the configured ceiling (default USER).
* **Injection quarantine** — if untrusted content carries an injection payload,
  it is not rejected outright (the SRE agent still needs to *see* the poisoned
  log), but it is forced to ``CANDIDATE`` status so it can never be retrieved as
  an active fact, and the payload is flagged in the returned report.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegismem.errors import GuardrailError
from aegismem.guardrails import pii
from aegismem.guardrails.injection import InjectionVerdict, PromptInjectionScanner
from aegismem.guardrails.trust import is_untrusted, trust_rank
from aegismem.mcp.gateway import ToolResult
from aegismem.memory.models import MemoryCreate, MemoryStatus, TrustLevel

# Source prefixes that mark an externally-originated (untrusted) write, regardless
# of the trust label the payload claims.
_UNTRUSTED_SOURCE_PREFIXES = ("tool:", "tool_result", "rag", "memory", "web", "log", "webhook")


def _untrusted_origin(data: MemoryCreate) -> bool:
    if is_untrusted(data.trust):
        return True
    src = data.source.lower()
    return any(src.startswith(p) for p in _UNTRUSTED_SOURCE_PREFIXES)


@dataclass(slots=True, frozen=True)
class WriteReport:
    """The sanitized write plus what the boundary did to it."""

    create: MemoryCreate
    trust_clamped: bool
    quarantined: bool
    injection: InjectionVerdict
    pii_kinds: list[str]


@dataclass(slots=True)
class MemoryWriteSanitizer:
    scanner: PromptInjectionScanner
    max_content_chars: int = 20000
    untrusted_write_max_trust: TrustLevel = TrustLevel.USER

    def sanitize(self, data: MemoryCreate) -> WriteReport:
        if len(data.content) > self.max_content_chars:
            raise GuardrailError(
                f"memory content exceeds limit "
                f"({len(data.content)} > {self.max_content_chars} chars)",
                stage="guardrails.memory_write",
            )

        patch: dict[str, object] = {}
        trust_clamped = False
        quarantined = False

        # Trust ceiling: an externally-originated write can never be stored at a
        # trust MORE privileged than the ceiling (default USER), even if the
        # payload claims SYSTEM/DEVELOPER. Lower rank = more privileged.
        untrusted_origin = _untrusted_origin(data)
        if untrusted_origin and trust_rank(data.trust) < trust_rank(
            self.untrusted_write_max_trust
        ):
            patch["trust"] = self.untrusted_write_max_trust
            trust_clamped = True

        verdict = self.scanner.scan(data.content)
        if verdict.blocked and untrusted_origin:
            # Never let injected untrusted content enter as active truth.
            if data.status not in (MemoryStatus.CANDIDATE, MemoryStatus.VALIDATING):
                patch["status"] = MemoryStatus.CANDIDATE
                quarantined = True
            elif data.status is MemoryStatus.VALIDATING:
                pass  # already pre-active; leave for the conflict resolver

        pii_kinds = [f.kind for f in pii.scan(data.content)]
        sanitized = data.model_copy(update=patch) if patch else data
        return WriteReport(
            create=sanitized,
            trust_clamped=trust_clamped,
            quarantined=quarantined,
            injection=verdict,
            pii_kinds=sorted(set(pii_kinds)),
        )


@dataclass(slots=True, frozen=True)
class ToolResultReport:
    result: ToolResult
    injection: InjectionVerdict
    contains_instructions: bool


@dataclass(slots=True)
class ToolResultValidator:
    """Tool output is TOOL_RESULT trust — untrusted data. It is scanned for
    embedded instructions so the boundary/agent treats it as data even when it
    contains an injection payload; it is never allowed to escalate trust."""

    scanner: PromptInjectionScanner
    max_result_chars: int = 32000

    def validate(self, result: ToolResult) -> ToolResultReport:
        if not is_untrusted(result.trust):  # defensive: tool output is untrusted
            raise GuardrailError(
                f"tool result claimed non-untrusted trust {result.trust.value!r}",
                stage="guardrails.tool_result",
            )
        text = result.output
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars]
            result = result.model_copy(update={"output": text, "truncated": True})
        verdict = self.scanner.scan(text)
        return ToolResultReport(
            result=result,
            injection=verdict,
            contains_instructions=verdict.blocked,
        )
