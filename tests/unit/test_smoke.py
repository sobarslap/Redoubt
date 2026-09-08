"""Stage 0 gate: the package imports, config loads, and the CLI runs."""

from __future__ import annotations

import aegismem
from aegismem.cli import main
from aegismem.config.settings import Settings, get_settings


def test_version_present() -> None:
    assert aegismem.__version__ == "0.0.0"


def test_settings_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.llm_provider in {"gemini", "claude", "ollama"}


def test_packaged_yaml_loads() -> None:
    settings = get_settings()
    budgets = settings.budgets()
    window = budgets["context_window"]
    # The allocator percentages must not overcommit the context window.
    assert sum(window.values()) <= 1.0 + 1e-9
    assert settings.policies()["retrieval"]["top_k"] >= 1
    assert "roles" in settings.models()


def test_cli_version_runs() -> None:
    assert main(["version"]) == 0
    assert main(["config"]) == 0
