"""AegisMem command-line entry point.

Ships ``version`` and ``config`` (Stage 0) and ``replay`` (Phase 7). Remaining
subcommands (``serve``, ``demo``) are added as their subsystems land.
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

    replay_p = sub.add_parser("replay", help="Reconstruct a run from the trace store")
    replay_p.add_argument("run_id", help="run_id to reconstruct (e.g. run_...)")
    replay_p.add_argument(
        "--trace-path", default="traces/trace.jsonl", help="path to the JSONL trace log"
    )
    replay_p.add_argument("--json", action="store_true", help="emit the reconstruction as JSON")

    args = parser.parse_args(argv)

    if args.command == "replay":
        return _replay(args.run_id, args.trace_path, as_json=args.json)

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


def _replay(run_id: str, trace_path: str, *, as_json: bool) -> int:
    from aegismem.errors import NotFoundError
    from aegismem.observability.replay import Replayer
    from aegismem.observability.trace_store import JSONLTraceStore

    replayer = Replayer(JSONLTraceStore(trace_path))
    try:
        run = replayer.reconstruct(run_id)
    except NotFoundError as exc:
        print(f"error: {exc.envelope.message}", file=sys.stderr)
        return 1

    if as_json:
        json.dump(run.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print(f"run {run.run_id}  request {run.request_id}  trace {run.trace_id}")
    print(f"status={run.status}  total={run.total_ms:.2f}ms  calls={run.calls}")
    print(f"input: {run.input[:120]}")
    print("pipeline:")
    for step in run.timeline:
        attrs = "  ".join(f"{k}={v}" for k, v in step.attributes.items())
        print(f"  {step.seq:>2}. {step.name:<14} {step.duration_ms:>8.2f}ms  "
              f"[{step.status}]  {attrs}")
    if run.outcome:
        print(f"outcome: {run.outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
