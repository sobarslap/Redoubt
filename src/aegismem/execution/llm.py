"""Provider-agnostic LLM client — the one seam every provider hides behind.

The runtime never imports a vendor SDK directly; it depends on :class:`LLMClient`.
Concrete adapters (Gemini, Claude, Ollama) are import-guarded so a clean checkout
runs without any of them, and :class:`MockProvider` gives a deterministic,
offline, keyless path used by the demo and the tests — the whole engine is
provable at zero cost.

A :class:`ResilientClient` wraps a primary with the failure policy
(``llm_request: retry_then_fallback``): bounded retries with backoff, then the
next provider in the fallback chain. Providers are resolved from
``config/models.yaml`` by role, so model routing is configuration, not code.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aegismem.context.tokens import HeuristicTokenCounter
from aegismem.errors import AegisError, ErrorCategory


class LLMError(AegisError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(
            "llm_error", ErrorCategory.LLM, message, retryable=retryable, stage="execution.llm"
        )


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation the model requested — executed only through the guarded
    Execute_Tool gateway, never directly. The model proposes; the gateway decides."""

    tool: str
    args: dict[str, object] = field(default_factory=dict)
    operation: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse: ...


_TOKEN_COUNTER = HeuristicTokenCounter()


def _est_tokens(text: str) -> int:
    # Reuse the runtime's single token heuristic so cost/usage estimates never
    # drift from the context-budget accounting.
    return max(1, _TOKEN_COUNTER.count(text))


@dataclass
class MockProvider:
    """Deterministic, offline LLM. No network, no key — the demo/tests run anywhere.

    ``scripts`` maps a prompt substring to a reply string. ``tool_plan`` maps a
    substring to tool calls the mock will request *once* (until the prompt shows
    those tools' results), so the agent's reason↔tool loop is exercised
    deterministically.
    """

    name: str = "mock"
    model: str = "mock-1"
    scripts: list[tuple[str, str]] = field(default_factory=list)
    tool_plan: list[tuple[str, tuple[ToolCall, ...]]] = field(default_factory=list)

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse:
        low = prompt.lower()
        for needle, calls in self.tool_plan:
            # Emit the tool calls only until their results are already in context.
            # Results are fenced as untrusted data labeled "tool:<name>" (see agent._reason).
            if needle.lower() in low and not all(f"tool:{c.tool}" in prompt for c in calls):
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    text="",
                    input_tokens=_est_tokens(system + prompt),
                    tool_calls=calls,
                )
        text = self._answer(low)
        return LLMResponse(
            text=text,
            input_tokens=_est_tokens(system + prompt),
            output_tokens=_est_tokens(text),
            provider=self.name,
            model=self.model,
        )

    def _answer(self, low: str) -> str:
        for needle, reply in self.scripts:
            if needle.lower() in low:
                return reply
        return "Acknowledged. I will use only the grounded facts provided and cite them."


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SpendError(AegisError):
    """Raised when a run exceeds its token or spend ceiling. Not retryable."""

    def __init__(self, message: str) -> None:
        super().__init__("spend_exceeded", ErrorCategory.BUDGET, message, stage="execution.llm")


# USD per 1M tokens (input, output). Rough public list prices; used only for the
# optional spend cap, and easy to update in one place.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "mock-1": (0.0, 0.0),
}


@dataclass
class CostGuard:
    """Per-run token/spend ceiling enforced across every LLM call in the run.

    ``charge`` accumulates a response's usage and raises :class:`SpendError` once a
    ceiling is crossed — a deterministic guard against runaway cost, checked at the
    boundary rather than hoped for."""

    max_tokens: int = 200_000
    max_usd: float | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    spent_usd: float = 0.0

    def charge(self, resp: LLMResponse) -> None:
        self.usage.input_tokens += resp.input_tokens
        self.usage.output_tokens += resp.output_tokens
        self.usage.calls += 1
        pin, pout = _PRICE_PER_MTOK.get(resp.model, (0.0, 0.0))
        self.spent_usd += (resp.input_tokens * pin + resp.output_tokens * pout) / 1_000_000
        if self.usage.total_tokens > self.max_tokens:
            raise SpendError(
                f"run token ceiling exceeded ({self.usage.total_tokens} > {self.max_tokens})"
            )
        if self.max_usd is not None and self.spent_usd > self.max_usd:
            raise SpendError(
                f"run spend ceiling exceeded (${self.spent_usd:.4f} > ${self.max_usd})"
            )


class GeminiProvider:  # pragma: no cover - requires network + key
    """Adapter for google-genai. Import-guarded; only used when configured."""

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        try:
            from google import genai  # type: ignore[import-not-found]
        except Exception as exc:
            raise LLMError(f"google-genai not installed: {exc}", retryable=False) from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse:
        try:
            full = f"{system}\n\n{prompt}" if system else prompt
            resp = self._client.models.generate_content(model=self.model, contents=full)
            text = getattr(resp, "text", "") or ""
        except Exception as exc:
            raise LLMError(f"gemini request failed: {exc}") from exc
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=_est_tokens(system + prompt),
            output_tokens=_est_tokens(text),
        )


class ClaudeProvider:  # pragma: no cover - requires network + key
    name = "claude"

    def __init__(self, model: str = "claude-opus-5", api_key: str | None = None) -> None:
        try:
            import anthropic  # type: ignore[import-not-found]
        except Exception as exc:
            raise LLMError(f"anthropic not installed: {exc}", retryable=False) from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse:
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content)
            usage = getattr(msg, "usage", None)
        except Exception as exc:
            raise LLMError(f"claude request failed: {exc}") from exc
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )


class OllamaProvider:  # pragma: no cover - requires local server
    name = "ollama"

    def __init__(self, model: str = "qwen2.5", host: str = "http://localhost:11434") -> None:
        self.model = model
        self._host = host

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse:
        try:
            import json
            import urllib.request

            payload = json.dumps(
                {"model": self.model, "prompt": prompt, "system": system, "stream": False}
            ).encode()
            req = urllib.request.Request(
                f"{self._host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            text = data.get("response", "")
        except Exception as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=_est_tokens(system + prompt),
            output_tokens=_est_tokens(text),
        )


class ResilientClient:
    """Retry-with-backoff on the primary, then fall back through the chain."""

    def __init__(
        self,
        primary: LLMClient,
        fallbacks: Sequence[LLMClient] = (),
        *,
        max_retries: int = 2,
        backoff_s: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._chain: list[LLMClient] = [primary, *fallbacks]
        self._max_retries = max_retries
        self._backoff = backoff_s
        self._sleep = sleep
        self.name = primary.name

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse:
        last: Exception | None = None
        for client in self._chain:
            for attempt in range(self._max_retries + 1):
                try:
                    return client.complete(prompt, system=system, max_tokens=max_tokens)
                except LLMError as exc:
                    last = exc
                    if not exc.envelope.retryable or attempt == self._max_retries:
                        break
                    self._sleep(self._backoff * (2**attempt))
        raise LLMError(f"all providers exhausted: {last}", retryable=False)
