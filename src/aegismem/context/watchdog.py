"""Token watchdog — fires compaction at the high watermark.

Uses exact token counts (not turn counts) against the allocated window. When
usage crosses the configured high watermark, a compaction run is triggered.
"""

from __future__ import annotations

from collections.abc import Iterable

from aegismem.context.state import ConversationTurn
from aegismem.context.tokens import TokenCounter


class TokenWatchdog:
    def __init__(self, counter: TokenCounter, window: int, high_watermark: float = 0.75) -> None:
        if not 0.0 < high_watermark <= 1.0:
            raise ValueError("high_watermark must be in (0, 1]")
        self._counter = counter
        self._window = window
        self._hwm = high_watermark

    def count_turns(self, turns: Iterable[ConversationTurn]) -> int:
        return sum(self._counter.count(t.content) for t in turns)

    def usage_fraction(self, used_tokens: int) -> float:
        return used_tokens / self._window if self._window else 0.0

    def should_compact(self, used_tokens: int) -> bool:
        return self.usage_fraction(used_tokens) >= self._hwm
