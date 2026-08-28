"""Compare human movement against the unchanged production safety/controller probe."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from brotato_ai.data.human_demo import _from_blob


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare human actions with controller recommendations")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(str(args.dataset))
    rows = connection.execute("SELECT action,controller_blob,derived_blob FROM frames").fetchall()
    connection.close()
    counts = Counter()
    for human_action, controller_blob, derived_blob in rows:
        controller = _from_blob(controller_blob, {})
        derived = _from_blob(derived_blob, {})
        safest = controller.get("safest_action")
        counts["frames"] += 1
        counts["human_matches_safest"] += int(safest is not None and int(human_action) == int(safest))
        counts["safest_action_exists"] += int(controller.get("safe_action_exists", False))
        counts["no_safe_action"] += int(controller.get("no_safe_action", False))
        counts["hazard_actionable"] += int(derived.get("hazard_actionable", False))
        counts["escape"] += int(derived.get("escape", False))
    total = max(1, counts["frames"])
    report = dict(counts)
    report.update({
        "human_matches_safest_rate": counts["human_matches_safest"] / total,
        "safe_action_available_rate": counts["safest_action_exists"] / total,
        "no_safe_action_rate": counts["no_safe_action"] / total,
        "hazard_actionable_rate": counts["hazard_actionable"] / total,
        "escape_time_fraction": counts["escape"] / total,
        "definition": "safest is the current shared hazard/recovery architecture evaluated offline; this is not a claim that a missing trained policy was reconstructed",
    })
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
