"""PII / secret detection and redaction.

Deterministic regex detectors for the credential and PII classes that show up in
SRE logs and webhook payloads (the demo's threat surface): emails, cloud API
keys, bearer tokens, private-key blocks, credit-card numbers (Luhn-checked to
cut false positives), and US SSNs. Used two ways:

* on input/tool-result ingestion, to flag and optionally redact secrets before
  they enter memory or context, and
* by the output filter, so detected secrets are scrubbed from responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class _Detector:
    kind: str
    regex: re.Pattern[str]
    luhn: bool = False


_DETECTORS: tuple[_Detector, ...] = (
    _Detector(
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
            r".*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    _Detector("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    _Detector("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    _Detector("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b")),
    _Detector(
        "generic_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}"
        ),
    ),
    _Detector("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    _Detector("ssn", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    _Detector("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), luhn=True),
)


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(slots=True, frozen=True)
class PiiFinding:
    kind: str
    value: str
    start: int
    end: int


def scan(text: str) -> list[PiiFinding]:
    findings: list[PiiFinding] = []
    for det in _DETECTORS:
        for m in det.regex.finditer(text):
            if det.luhn and not _luhn_ok(m.group(0)):
                continue
            findings.append(PiiFinding(det.kind, m.group(0), m.start(), m.end()))
    return findings


def redact(text: str, *, findings: list[PiiFinding] | None = None) -> str:
    """Replace detected PII/secrets with ``[REDACTED:kind]`` markers."""
    found = findings if findings is not None else scan(text)
    # Replace right-to-left so offsets stay valid.
    for f in sorted(found, key=lambda x: x.start, reverse=True):
        text = text[: f.start] + f"[REDACTED:{f.kind}]" + text[f.end :]
    return text
