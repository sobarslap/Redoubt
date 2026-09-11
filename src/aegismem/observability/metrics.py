"""Prometheus metrics — the service's operational signals (Production P7).

A tiny, dependency-free registry that renders the Prometheus text exposition
format, scraped at ``/metrics``. Counters track runs, failures, guardrail blocks,
and tool refusals; a summary tracks run latency. Alert rules in
``ops/alerts.yml`` fire on these (a risen attack-success-rate or a blown P95 pages
someone), closing the loop the plan names: the CI gates become live alerts.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsRegistry:
    counters: dict[str, float] = field(default_factory=dict)
    summaries: dict[str, list[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.summaries.setdefault(name, []).append(value)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, val in sorted(self.counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")
            for name, obs in sorted(self.summaries.items()):
                count = len(obs)
                total = sum(obs)
                ordered = sorted(obs)
                p50 = statistics.median(ordered) if obs else 0.0
                p95 = ordered[max(0, int(0.95 * count) - 1)] if obs else 0.0
                lines.append(f"# TYPE {name} summary")
                lines.append(f'{name}{{quantile="0.5"}} {p50}')
                lines.append(f'{name}{{quantile="0.95"}} {p95}')
                lines.append(f"{name}_count {count}")
                lines.append(f"{name}_sum {total}")
        return "\n".join(lines) + "\n"
