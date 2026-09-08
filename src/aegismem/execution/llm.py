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

from aegismem.errors import AegisError, ErrorCategory


class LLMError(AegisError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(
            "llm_error", ErrorCategory.LLM, message, retryable=retryable, stage="execution.llm"
        )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse: ...


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class MockProvider:
    """Deterministic, offline LLM. Scripted responses matched by substring; falls
    back to a grounded echo. No network, no key — the demo/tests run anywhere."""

    name: str = "mock"
    model: str = "mock-1"
    scripts: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 512) -> LLMResponse:
        text = self._answer(prompt)
        return LLMResponse(
            text=text,
            input_tokens=_est_tokens(system + prompt),
            output_tokens=_est_tokens(text),
            provider=self.name,
            model=self.model,
        )

    def _answer(self, prompt: str) -> str:
        low = prompt.lower()
        for needle, reply in self.scripts:
            if needle.lower() in low:
                return reply
        return "Acknowledged. I will use only the grounded facts provided and cite them."


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
