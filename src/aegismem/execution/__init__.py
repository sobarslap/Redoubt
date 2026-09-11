"""AegisMem execution subsystem — the agent loop and the LLM abstraction.

``AgentRuntime`` wires the deterministic execution contract end to end; the
provider-agnostic ``LLMClient`` (with an offline ``MockProvider`` and a resilient
retry/fallback wrapper) keeps every vendor SDK behind one seam.
"""

from aegismem.execution.agent import AgentRuntime
from aegismem.execution.factory import ProviderConfigError, build_client
from aegismem.execution.llm import (
    ClaudeProvider,
    CostGuard,
    GeminiProvider,
    LLMClient,
    LLMError,
    LLMResponse,
    LLMUsage,
    MockProvider,
    OllamaProvider,
    ResilientClient,
    SpendError,
    ToolCall,
)

__all__ = [
    "AgentRuntime",
    "ClaudeProvider",
    "CostGuard",
    "GeminiProvider",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LLMUsage",
    "MockProvider",
    "OllamaProvider",
    "ProviderConfigError",
    "ResilientClient",
    "SpendError",
    "ToolCall",
    "build_client",
]
