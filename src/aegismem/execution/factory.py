"""Provider factory — resolve a role to a concrete, resilient LLM client.

Model routing is configuration, not code: ``config/models.yaml`` maps roles
(workhorse / reasoning / judge / offline) to a provider + model, and a
`fallback_chain` orders providers for the failure policy. This factory reads that
config, pulls API keys from the environment (never from disk or code), constructs
the adapter, and wraps it in :class:`ResilientClient` with the remaining chain as
fallbacks.

Keyless by default: if the resolved provider has no key configured, the factory
returns the offline :class:`MockProvider` unless ``strict=True``, in which case it
raises a typed error naming the missing variable. This is what lets the whole
runtime — demo and tests — run at zero cost while a real deployment (Production
Phase P1) sets one env var to go live.
"""

from __future__ import annotations

import os

from aegismem.config.settings import Settings, get_settings
from aegismem.errors import AegisError, ErrorCategory
from aegismem.execution.llm import (
    ClaudeProvider,
    GeminiProvider,
    LLMClient,
    MockProvider,
    OllamaProvider,
    ResilientClient,
)

# Env var that holds each provider's key. Ollama is local and needs none.
_KEY_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "claude": ("ANTHROPIC_API_KEY",),
    "ollama": (),
}


class ProviderConfigError(AegisError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "provider_config", ErrorCategory.INTERNAL, message, stage="execution.factory"
        )


def _key_for(provider: str) -> str | None:
    for var in _KEY_ENV.get(provider, ()):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _has_key(provider: str) -> bool:
    return provider == "ollama" or _key_for(provider) is not None


def _build_provider(provider: str, model: str) -> LLMClient:
    if provider == "gemini":
        return GeminiProvider(model=model, api_key=_key_for("gemini"))
    if provider == "claude":
        return ClaudeProvider(model=model, api_key=_key_for("claude"))
    if provider == "ollama":
        return OllamaProvider(model=model)
    if provider == "mock":
        return MockProvider(model=model)
    raise ProviderConfigError(f"unknown provider {provider!r}")


def build_client(
    role: str = "workhorse",
    *,
    settings: Settings | None = None,
    strict: bool = False,
) -> LLMClient:
    """Resolve ``role`` to a resilient client (primary + fallback chain).

    ``strict=True`` refuses to silently fall back to the offline mock when a real
    provider is configured but unkeyed — use it in a deployment where a missing
    key must fail loudly.
    """
    models = (settings or get_settings()).models()
    roles = models.get("roles", {})
    if role not in roles:
        raise ProviderConfigError(f"role {role!r} not in models.yaml roles")
    provider = str(roles[role]["provider"])
    model = str(roles[role]["model"])

    if not _has_key(provider):
        if strict:
            envs = " or ".join(_KEY_ENV.get(provider, ("<none>",))) or "<none>"
            raise ProviderConfigError(
                f"provider {provider!r} for role {role!r} has no key (set {envs})"
            )
        return MockProvider(model=f"mock-1({provider}:{model})")

    primary = _build_provider(provider, model)

    # Fallback chain: the configured order, minus the primary, keyed providers only.
    chain: list[LLMClient] = []
    for fb in models.get("fallback_chain", []):
        if fb != provider and _has_key(fb):
            fb_model = _model_for_provider(roles, fb) or model
            chain.append(_build_provider(fb, fb_model))
    return ResilientClient(primary, chain)


def _model_for_provider(roles: dict[str, object], provider: str) -> str | None:
    for spec in roles.values():
        if isinstance(spec, dict) and spec.get("provider") == provider:
            return str(spec.get("model"))
    return None
