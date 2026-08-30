"""Validate a human demonstration dataset and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brotato_ai.data.human_demo import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a human demonstration SQLite dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--require-capture",
        action="store_true",
        help="fail unless raw streams, reward/features/outcomes, and episode boundaries are complete",
    )
    args = parser.parse_args()
    report = validate_dataset(args.dataset, require_capture=args.require_capture)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
