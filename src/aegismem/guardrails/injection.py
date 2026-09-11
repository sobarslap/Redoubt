"""Prompt-injection / instruction-hijack detector — defense-in-depth.

Deterministic-first: a weighted regex taxonomy over a documented set of attack
families (named after the AI-Infra-Guard taxonomy the red-team suite scores
against). This is *not* the primary defense — the structural trust boundary in
``trust.py`` is — but it flags injected instructions so the boundary can refuse to
act on them, route poisoned memory writes to review, and drive the security
benchmark's attack-success-rate.

Fail-closed: a scan that raises internally is reported as blocked, never allowed.

The detector reports *which* families matched so findings are auditable and the
red-team suite can assert coverage of each named attack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel


class AttackFamily:
    """AI-Infra-Guard-aligned family names (stable identifiers for scoring)."""

    INSTRUCTION_OVERRIDE = "instruction_override"  # "ignore previous instructions"
    FAKE_SYSTEM = "fake_system_instruction"  # forged SYSTEM/role turn
    ROLE_HIJACK = "role_hijack"  # ActorAttack / persona reassignment
    EXFILTRATION = "exfiltration"  # send/email/leak secrets
    TOOL_HIJACK = "tool_hijack"  # command / tool-call injection
    MEMORY_POISON = "memory_poisoning"  # rewrite/forget stored facts
    SECRET_PROBE = "secret_probe"  # reveal system prompt / keys
    MANY_SHOT = "many_shot_jailbreak"  # stacked fake dialogue turns
    OBFUSCATION = "obfuscation"  # base64/hex/"decode and run"


@dataclass(slots=True, frozen=True)
class _Pattern:
    family: str
    weight: float
    regex: re.Pattern[str]


def _p(family: str, weight: float, pattern: str) -> _Pattern:
    return _Pattern(family, weight, re.compile(pattern, re.IGNORECASE))


# Each pattern carries a weight; a single high-weight hit is enough to block.
_PATTERNS: tuple[_Pattern, ...] = (
    _p(
        AttackFamily.INSTRUCTION_OVERRIDE,
        0.9,
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
    ),
    _p(
        AttackFamily.INSTRUCTION_OVERRIDE,
        0.9,
        r"\bdisregard\s+(?:all\s+)?(?:previous|prior|the\s+above|your)\s+\w+",
    ),
    _p(
        AttackFamily.INSTRUCTION_OVERRIDE,
        0.7,
        r"\b(?:forget|override)\s+(?:everything|your\s+(?:rules|instructions|guidelines))",
    ),
    _p(AttackFamily.INSTRUCTION_OVERRIDE, 0.6, r"\bnew\s+instructions?\s*:"),
    _p(AttackFamily.FAKE_SYSTEM, 0.9, r"(?m)^\s*(?:system|developer|assistant)\s*:\s*\S"),
    _p(AttackFamily.FAKE_SYSTEM, 0.85, r"</?(?:system|system_prompt|im_start|im_end)\b"),
    _p(AttackFamily.FAKE_SYSTEM, 0.7, r"\bthis\s+is\s+(?:an?\s+)?(?:official|admin|system)\s+"),
    _p(AttackFamily.ROLE_HIJACK, 0.85, r"\byou\s+are\s+now\s+(?:a|an|the|no\s+longer)\b"),
    _p(AttackFamily.ROLE_HIJACK, 0.8, r"\b(?:act|behave|roleplay|pretend)\s+as\s+(?:if\s+)?"),
    _p(AttackFamily.ROLE_HIJACK, 0.8, r"\b(?:DAN|developer\s+mode|jailbreak\s+mode)\b"),
    _p(
        AttackFamily.EXFILTRATION,
        0.9,
        r"\b(?:email|send|post|upload|exfiltrate|leak|transmit)\b.{0,40}"
        r"\b(?:credential|password|secret|api[_\s-]?key|token|\.env|private\s+key)",
    ),
    _p(AttackFamily.EXFILTRATION, 0.8, r"\b(?:send|post|curl|fetch|GET|POST)\b.{0,30}https?://"),
    _p(
        AttackFamily.TOOL_HIJACK,
        0.85,
        r"\b(?:call|invoke|execute|run)\s+(?:the\s+)?(?:tool\s+)?"
        r"(?:restart_service|delete_|drop_|shutdown|rm\s+-rf|Execute_Tool)",
    ),
    _p(
        AttackFamily.TOOL_HIJACK,
        0.8,
        r"\bmark\b.{0,30}\b(?:compromised|breached|deleted|resolved)\b",
    ),
    _p(
        AttackFamily.MEMORY_POISON,
        0.85,
        r"\b(?:remember|store|save|record)\s+(?:that\s+)?.{0,40}"
        r"\b(?:always|forever|permanently|as\s+(?:a\s+)?fact)",
    ),
    _p(
        AttackFamily.MEMORY_POISON,
        0.8,
        r"\b(?:update|overwrite|replace|delete|forget)\s+(?:your\s+)?(?:memory|the\s+fact|all\s+memories)",
    ),
    _p(
        AttackFamily.SECRET_PROBE,
        0.9,
        r"\b(?:reveal|show|print|repeat|output|leak)\b.{0,30}"
        r"\b(?:system\s+prompt|instructions|rules|api[_\s-]?key|secret|password)",
    ),
    _p(
        AttackFamily.SECRET_PROBE,
        0.7,
        r"\bwhat\s+(?:are|were)\s+your\s+(?:original\s+)?(?:instructions|rules|system\s+prompt)",
    ),
    _p(
        AttackFamily.OBFUSCATION,
        0.7,
        r"\b(?:base64|rot13|hex)\b.{0,40}\b(?:decode|execute|run|eval)",
    ),
    _p(
        AttackFamily.OBFUSCATION,
        0.6,
        r"\b(?:decode|deobfuscate)\b.{0,20}\bthen\b.{0,20}\b(?:run|execute)",
    ),
)

# Many-shot: N stacked fake dialogue turns is itself the attack.
_TURN_RE = re.compile(r"(?mi)^\s*(?:user|human|assistant|Q|A)\s*:")
_MANY_SHOT_MIN_TURNS = 6


class InjectionVerdict(BaseModel):
    blocked: bool
    score: float
    families: list[str] = []
    matches: list[str] = []


@dataclass(slots=True)
class PromptInjectionScanner:
    block_threshold: float = 0.5
    _patterns: tuple[_Pattern, ...] = field(default=_PATTERNS)

    def scan(self, text: str) -> InjectionVerdict:
        try:
            return self._scan(text)
        except Exception:  # fail-closed: an unscannable input is treated as hostile
            return InjectionVerdict(blocked=True, score=1.0, families=["scanner_error"], matches=[])

    def _scan(self, text: str) -> InjectionVerdict:
        families: dict[str, float] = {}
        matches: list[str] = []
        for pat in self._patterns:
            m = pat.regex.search(text)
            if m:
                families[pat.family] = max(families.get(pat.family, 0.0), pat.weight)
                snippet = m.group(0)
                matches.append(snippet if len(snippet) <= 80 else snippet[:77] + "...")

        turns = len(_TURN_RE.findall(text))
        if turns >= _MANY_SHOT_MIN_TURNS:
            families[AttackFamily.MANY_SHOT] = max(
                families.get(AttackFamily.MANY_SHOT, 0.0),
                min(1.0, 0.5 + 0.1 * (turns - _MANY_SHOT_MIN_TURNS)),
            )

        score = max(families.values(), default=0.0)
        return InjectionVerdict(
            blocked=score >= self.block_threshold,
            score=round(score, 4),
            families=sorted(families),
            matches=matches[:10],
        )
