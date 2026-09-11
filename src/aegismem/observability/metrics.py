"""Prometheus metrics — the service's operational signals (Production P7).

A tiny, dependency-free registry that renders the Prometheus text exposition
format, scraped at ``/metrics``. Counters track runs, failures, guardrail blocks,
and tool refusals; a summary tracks run latency. Alert rules in
``ops/alerts.yml`` fire on these (a risen attack-success-rate or a blown P95 pages
someone), closing the loop the plan names: the CI gates become live alerts.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

# Quantiles are computed over a bounded window of recent observations so a
# long-running service cannot leak memory; count/sum stay exact.
_MAX_RECENT = 4096


@dataclass
class _Summary:
    count: int = 0
    total: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT))


@dataclass
class MetricsRegistry:
    counters: dict[str, float] = field(default_factory=dict)
    summaries: dict[str, _Summary] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            s = self.summaries.setdefault(name, _Summary())
            s.count += 1
            s.total += value
            s.recent.append(value)  # bounded by maxlen

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, val in sorted(self.counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")
            for name, s in sorted(self.summaries.items()):
                count = s.count
                total = s.total
                ordered = sorted(s.recent)
                p50 = statistics.median(ordered) if ordered else 0.0
                p95 = ordered[max(0, int(0.95 * len(ordered)) - 1)] if ordered else 0.0
                lines.append(f"# TYPE {name} summary")
                lines.append(f'{name}{{quantile="0.5"}} {p50}')
                lines.append(f'{name}{{quantile="0.95"}} {p95}')
                lines.append(f"{name}_count {count}")
                lines.append(f"{name}_sum {total}")
        return "\n".join(lines) + "\n"
