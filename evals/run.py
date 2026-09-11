"""Golden-dataset runner. ``python -m evals.run`` (exit 1 on any gate breach).

This is the CI eval gate: a dropped per-category pass-rate, or a nonzero
memory-corruption / unauthorized-tool-exec / attack-success count, fails the run.
"""

from __future__ import annotations

import sys

from evals.harness import run_dataset


def main() -> int:
    report = run_dataset()
    print(f"golden dataset: {report.passed}/{report.total} scenarios passed")
    print("by category:")
    for c in report.categories:
        flag = "ok " if c.ok else "FAIL"
        print(
            f"  [{flag}] {c.category.value:<17} {c.passed}/{c.total}  "
            f"(pass_rate={c.pass_rate:.2f} gate={c.gate:.2f})"
        )
    print("invariants:")
    print(f"  memory_corruption      {report.memory_corruption}  (target 0)")
    print(f"  unauthorized_tool_exec {report.unauthorized_tool_exec}  (target 0)")
    print(f"  attack_success         {report.attack_success}  (target 0)")

    if not report.ok:
        print("\nFAILURES:")
        for f in report.failures:
            print(f"  - {f}")
        print("\nEVAL GATE: FAIL")
        return 1
    print("\nEVAL GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
