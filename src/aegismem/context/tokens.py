"""Token counting — exact where a real tokenizer is present, honest fallback else.

``TokenCounter`` is the seam. Production uses provider token APIs or ``tiktoken``;
:class:`HeuristicTokenCounter` is the zero-dependency default (≈4 chars/token,
never < the word count) so the context subsystem measures budgets on any
interpreter without a tokenizer install. Counts are *tokens*, not turns — the
whole point of the watchdog is that it triggers on real size, not message count.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Deterministic char/word heuristic; a safe upper-ish estimate of tokens."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self._cpt = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        by_chars = round(len(text) / self._cpt)
        by_words = len(text.split())
        return max(1, by_chars, by_words)


def try_tiktoken_counter(encoding: str = "cl100k_base") -> TokenCounter | None:
    """Return a tiktoken-backed counter if the optional dependency is installed."""
    try:
        import tiktoken
    except ImportError:
        return None

    enc = tiktoken.get_encoding(encoding)

    class _Tik:
        def count(self, text: str) -> int:
            # Encode special-token strings (e.g. "<|endoftext|>") as ordinary text
            # instead of raising — conversation content is untrusted input.
            return len(enc.encode(text, disallowed_special=()))

    return _Tik()
