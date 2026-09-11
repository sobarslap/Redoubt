"""Load-test rig for the AegisMem service (Production Phase P5).

Drives the running FastAPI service with concurrent run requests to measure
throughput, tail latency, and the saturation point. Not imported by the test
suite (locust is an optional `bench`-group dependency); run against a live
service:

    uv sync --group bench
    uvicorn "aegismem.api.app:create_app" --factory --port 8000 &
    locust -f benchmarks/scale/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 10 --run-time 2m --headless

Set AEGISMEM_API_KEY to exercise the authenticated path.
"""

from __future__ import annotations

import os
import random

try:
    from locust import HttpUser, between, task
except Exception:  # pragma: no cover - locust is optional
    HttpUser = object  # type: ignore[assignment,misc]

    def task(fn):  # type: ignore[no-redef]
        return fn

    def between(a, b):  # type: ignore[no-redef]
        return None


_INPUTS = (
    "why is the checkout service erroring?",
    "how do we recover from connection pool exhaustion?",
    "what database did we migrate to?",
    "give the remediation for the incident",
)
_KEY = os.environ.get("AEGISMEM_API_KEY")


class RunUser(HttpUser):  # type: ignore[misc,valid-type]
    wait_time = between(0.1, 1.0)

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": _KEY} if _KEY else {}

    @task(4)
    def sync_run(self) -> None:
        self.client.post(  # type: ignore[attr-defined]
            "/runs",
            json={"session_id": "load", "input": random.choice(_INPUTS)},
            headers=self._headers(),
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/healthz")  # type: ignore[attr-defined]
