"""Post-deploy smoke test (Production Phase P8).

Hits a running service's health, run, and metrics endpoints; exits non-zero on
any failure so the deploy pipeline can gate promotion (deploy staging -> smoke ->
promote). No test framework, just the stdlib, so it runs in a minimal container.

    python -m ops.smoke http://localhost:8000
"""

from __future__ import annotations

import json
import sys
import urllib.request


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode()


def _post(url: str, payload: dict[str, object]) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode()


def main(base: str) -> int:
    checks: list[tuple[str, bool]] = []

    status, _ = _get(f"{base}/healthz")
    checks.append(("healthz 200", status == 200))

    status, _ = _get(f"{base}/readyz")
    checks.append(("readyz 200", status == 200))

    status, body = _post(f"{base}/runs", {"session_id": "smoke", "input": "ping"})
    checks.append(("run 200", status == 200))
    checks.append(("run has output", '"output"' in body))

    status, metrics = _get(f"{base}/metrics")
    checks.append(("metrics 200", status == 200))
    checks.append(("metrics has runs", "aegismem_runs_total" in metrics))

    ok = True
    for name, passed in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("SMOKE: PASS" if ok else "SMOKE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    sys.exit(main(base.rstrip("/")))
