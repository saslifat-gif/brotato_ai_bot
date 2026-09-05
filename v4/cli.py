"""Unified cross-platform CLI entrypoint for Brotato AI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _run_module(module_name: str, args: list[str]) -> int:
    import importlib

    module = importlib.import_module(module_name)
    main_func = getattr(module, "main", None)
    if not callable(main_func):
        print(f"[brotato-ai] Error: module {module_name} does not define main()", file=sys.stderr)
        return 1
    sys.argv = [module_name] + args
    try:
        result = main_func()
        return int(result) if result is not None else 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brotato-ai",
        description="Brotato AI V4 unified command-line tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # install-mod
    subparsers.add_parser("install-mod", help="Install the Godot RL bridge mod into Brotato")

    # record-human
    subparsers.add_parser("record-human", help="Record manual human gameplay demonstrations with F9 bookmarks")

    # validate-demo
    subparsers.add_parser("validate-demo", help="Validate human demonstration dataset SQLite files")

    # report-demo
    subparsers.add_parser("report-demo", help="Generate demonstration dataset quality report")

    # train-bc
    subparsers.add_parser("train-bc", help="Train event-based behavioral cloning human model")

    # train-rl
    subparsers.add_parser("train-rl", help="Train PPO reinforcement learning agent")

    # run-frozen / run-shadow
    subparsers.add_parser("run-frozen", help="Run frozen policy evaluation (model, bc, semantic, teacher)")
    subparsers.add_parser("run-shadow", help="Alias for run-frozen with shadow human logging")

    # dagger
    subparsers.add_parser("dagger", help="Run offline DAgger review and corrective labeling")

    # report-shadow
    subparsers.add_parser("report-shadow", help="Summarize and compare frozen/shadow evaluation results")

    # diagnose
    subparsers.add_parser("diagnose", help="Run bridge connectivity diagnostics")

    return parser


COMMAND_MAP = {
    "install-mod": "v4.install_mod",
    "record-human": "v4.record_human_demo",
    "validate-demo": "v4.validate_human_demo",
    "report-demo": "v4.report_human_demo_quality",
    "train-bc": "v4.train_event_human_bc",
    "train-rl": "v4.train",
    "run-frozen": "v4.run_frozen",
    "run-shadow": "v4.run_frozen",
    "dagger": "v4.dagger_corrective",
    "report-shadow": "v4.report_shadow",
    "diagnose": "v4.diagnose_bridge",
}


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args:
        build_parser().print_help()
        return 0

    command = raw_args[0]
    if command in {"-h", "--help"}:
        build_parser().print_help()
        return 0

    if command not in COMMAND_MAP:
        print(f"[brotato-ai] Unknown command: {command!r}. Use --help to list available commands.", file=sys.stderr)
        return 1

    module_name = COMMAND_MAP[command]
    if command == "run-shadow":
        previous_mode = os.environ.get("BROTATO_V4_POLICY_MODE")
        os.environ["BROTATO_V4_POLICY_MODE"] = "SHADOW_HUMAN"
        try:
            return _run_module(module_name, raw_args[1:])
        finally:
            if previous_mode is None:
                os.environ.pop("BROTATO_V4_POLICY_MODE", None)
            else:
                os.environ["BROTATO_V4_POLICY_MODE"] = previous_mode
    return _run_module(module_name, raw_args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
