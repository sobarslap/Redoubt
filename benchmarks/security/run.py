"""Security benchmark — attack-success-rate and false-positive rate.

Run: ``uv run python -m benchmarks.security.run``

Replays the red-team corpus through the injection scanner and reports, per named
attack family, how many payloads were caught, plus the aggregate
attack-success-rate (target 0) and the measured false-positive rate on benign SRE
chatter. These are the reproducible numbers behind ``docs/SECURITY.md`` and the
CI gate that fails on any rise in attack-success-rate.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from tests.adversarial.corpus import ATTACKS, BENIGN

from aegismem.guardrails.injection import PromptInjectionScanner

_RESULTS = Path(__file__).parent / "results"


def main() -> None:
    scanner = PromptInjectionScanner()

    per_family_total: dict[str, int] = defaultdict(int)
    per_family_caught: dict[str, int] = defaultdict(int)
    successes: list[str] = []

    for atk in ATTACKS:
        per_family_total[atk.family] += 1
        if scanner.scan(atk.payload).blocked:
            per_family_caught[atk.family] += 1
        else:
            successes.append(atk.name)

    caught = len(ATTACKS) - len(successes)
    asr = len(successes) / len(ATTACKS)
    false_positives = [t for t in BENIGN if scanner.scan(t).blocked]
    fpr = len(false_positives) / len(BENIGN)

    print(f"attacks={len(ATTACKS)} caught={caught}")
    print(f"attack_success_rate  {asr:.3f}  (target 0.000)")
    print(f"false_positive_rate  {fpr:.3f}  (benign={len(BENIGN)})")
    print("per-family coverage:")
    for fam in sorted(per_family_total):
        print(f"  {fam:<26} {per_family_caught[fam]}/{per_family_total[fam]}")
    if successes:
        print(f"UNCAUGHT: {successes}")

    _RESULTS.mkdir(exist_ok=True)
    out = _RESULTS / "security.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["attack_success_rate", f"{asr:.4f}"])
        w.writerow(["false_positive_rate", f"{fpr:.4f}"])
        w.writerow(["attacks_total", len(ATTACKS)])
        w.writerow(["attacks_caught", caught])
        for fam in sorted(per_family_total):
            w.writerow([f"family:{fam}", f"{per_family_caught[fam]}/{per_family_total[fam]}"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
