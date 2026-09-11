"""Optional ML prompt-injection classifier — defense-in-depth (Production P6).

The deterministic regex scanner in ``injection.py`` is the always-on gate; this
adds a small HF classifier (`deberta-v3-base-prompt-injection`) as a second
opinion, import-guarded so a clean checkout and keyless CI never load transformers
or torch. `CombinedScanner` runs both and blocks if *either* fires — the ML model
catches paraphrased attacks the regexes miss, the regexes stay fail-closed and
free. Neither is the primary defense: the structural trust boundary is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegismem.guardrails.injection import InjectionVerdict, PromptInjectionScanner


@dataclass
class HFInjectionClassifier:
    """`transformers` text-classification adapter. Lazy-loads on first scan; a
    load/inference failure fails closed (treated as an injection)."""

    model_name: str = "protectai/deberta-v3-base-prompt-injection"
    block_threshold: float = 0.5
    _pipe: object | None = field(default=None, repr=False)

    def _ensure(self) -> object:
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.model_name, truncation=True)
        return self._pipe

    def score(self, text: str) -> float:
        try:
            pipe = self._ensure()
            out = pipe(text[:2000])  # type: ignore[operator]
            row = out[0]
            label = str(row.get("label", "")).upper()
            prob = float(row.get("score", 0.0))
            # Model labels the injection class as INJECTION/LABEL_1.
            return prob if label in {"INJECTION", "LABEL_1", "1"} else 1.0 - prob
        except Exception:
            return 1.0  # fail-closed


@dataclass
class CombinedScanner:
    """Regex scanner OR ML classifier — blocks if either does."""

    regex: PromptInjectionScanner = field(default_factory=PromptInjectionScanner)
    ml: HFInjectionClassifier | None = None

    def scan(self, text: str) -> InjectionVerdict:
        verdict = self.regex.scan(text)
        if self.ml is None:
            return verdict
        ml_score = self.ml.score(text)
        blocked = verdict.blocked or ml_score >= self.ml.block_threshold
        families = list(verdict.families)
        if ml_score >= self.ml.block_threshold and "ml_classifier" not in families:
            families.append("ml_classifier")
        return InjectionVerdict(
            blocked=blocked,
            score=max(verdict.score, round(ml_score, 4)),
            families=sorted(families),
            matches=verdict.matches,
        )
