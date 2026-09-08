"""Runtime settings and YAML config loading.

Settings come from three layers, lowest precedence first:

1. Packaged YAML defaults (``budgets.yaml``, ``policies.yaml``, ``models.yaml``).
2. Environment variables prefixed ``AEGISMEM_``.
3. Explicit overrides passed at construction (tests, per-run budget overrides).

Only standard-library + PyYAML + pydantic are used here so the config layer has
no heavy dependencies and can be imported by any subsystem.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).parent


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / name
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


class Settings(BaseSettings):
    """Top-level runtime settings.

    Subsystem-specific policy (budgets, retrieval thresholds, tool allow/deny
    lists) lives in the YAML files and is loaded on demand rather than being
    flattened here, so policy can evolve without touching this schema.
    """

    model_config = SettingsConfigDict(
        env_prefix="AEGISMEM_",
        env_file=".env",
        extra="ignore",
    )

    environment: str = Field(default="dev", description="dev | test | prod")
    db_path: str = Field(default="aegismem.db", description="SQLite source-of-truth path")
    llm_provider: str = Field(default="gemini", description="gemini | claude | ollama")

    def budgets(self) -> dict[str, Any]:
        return _load_yaml("budgets.yaml")

    def policies(self) -> dict[str, Any]:
        return _load_yaml("policies.yaml")

    def models(self) -> dict[str, Any]:
        return _load_yaml("models.yaml")


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
