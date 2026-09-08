"""AegisMem guardrails subsystem — the security trust boundary (Phase 6).

External content is data, never instructions. Defenses are layered:

* ``trust``   — trust labels + structural, fenced context assembly (privilege
  separation + prompt hardening). The primary defense.
* ``injection`` — a deterministic injection/hijack detector mapped to named
  attack families (defense-in-depth).
* ``pii``     — secret/PII detection and redaction.
* ``sanitize`` — memory-write and tool-result validation (untrusted-ingest
  choke point; upholds memory_corruption = 0).
* ``output``  — outbound leakage filter.
* ``boundary`` — the ``SecurityBoundary`` façade the runtime contract calls.
"""

from aegismem.guardrails.boundary import InputDecision, SecurityBoundary
from aegismem.guardrails.injection import (
    AttackFamily,
    InjectionVerdict,
    PromptInjectionScanner,
)
from aegismem.guardrails.output import OutputVerdict, filter_output
from aegismem.guardrails.pii import PiiFinding, redact, scan
from aegismem.guardrails.sanitize import (
    MemoryWriteSanitizer,
    ToolResultValidator,
    WriteReport,
)
from aegismem.guardrails.trust import (
    IMMUTABLE_RULES,
    Segment,
    assemble_context,
    is_untrusted,
    trust_rank,
)

__all__ = [
    "IMMUTABLE_RULES",
    "AttackFamily",
    "InjectionVerdict",
    "InputDecision",
    "MemoryWriteSanitizer",
    "OutputVerdict",
    "PiiFinding",
    "PromptInjectionScanner",
    "SecurityBoundary",
    "Segment",
    "ToolResultValidator",
    "WriteReport",
    "assemble_context",
    "filter_output",
    "is_untrusted",
    "redact",
    "scan",
    "trust_rank",
]
