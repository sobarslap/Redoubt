"""AegisMem command-line entry point.

Stage 0 ships only ``version`` and ``config`` so the package is runnable and the
console-script wiring is verified end-to-end. Subcommands (``serve``, ``replay``,
``demo``) are added as their subsystems land.
"""

from __future__ import annotations

import argparse
import json
import sys

from aegismem import __version__
from aegismem.config.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aegismem", description="AegisMem agent runtime")
    parser.add_argument("--version", action="version", version=f"aegismem {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="Print the version and exit")
    sub.add_parser("config", help="Print resolved settings and loaded policy as JSON")

    args = parser.parse_args(argv)

    if args.command == "config":
        settings = get_settings()
        payload = {
            "settings": settings.model_dump(),
            "budgets": settings.budgets(),
            "policies": settings.policies(),
            "models": settings.models(),
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    # Default / "version": keep it simple.
    print(f"aegismem {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
