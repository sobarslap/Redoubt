"""AegisMem — a framework-free, production-grade agent runtime in pure Python.

The public surface is intentionally small at this stage; subsystems are wired in
phase by phase per the build plan (memory → retrieval → context → mcp →
guardrails → execution → observability). See ``ARCHITECTURE.md``.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
