"""Output validation — leakage filter on the response side of the contract.

Two jobs on every outbound response:

1. **Secret leakage**: scrub any known secret string (system-prompt fragments,
   API keys, tokens the run was seeded with) that the model may have echoed.
2. **PII leakage**: redact PII/credentials the model surfaced from untrusted
   content, unless the caller explicitly permits it.

Returns the filtered text plus a flag so the caller/trace can record that a leak
was caught (a caught leak is a security event worth surfacing, not silently
hidden).
"""

from __future__ import annotations

from dataclasses import dataclass

import aegismem.guardrails.pii as pii


@dataclass(slots=True, frozen=True)
class OutputVerdict:
    text: str
    leaked: bool
    reasons: list[str]


def filter_output(
    text: str,
    *,
    secrets: frozenset[str] = frozenset(),
    redact_pii: bool = True,
) -> OutputVerdict:
    reasons: list[str] = []
    out = text

    # Redact longer secrets first: a shorter secret that is a prefix/substring of a
    # longer one would otherwise be replaced first and leave part of the longer one.
    for secret in sorted(secrets, key=len, reverse=True):
        if secret and secret in out:
            out = out.replace(secret, "[REDACTED:secret]")
            reasons.append("known_secret")

    if redact_pii:
        findings = pii.scan(out)
        if findings:
            out = pii.redact(out, findings=findings)
            reasons.extend(f"pii:{f.kind}" for f in findings)

    return OutputVerdict(text=out, leaked=bool(reasons), reasons=sorted(set(reasons)))
