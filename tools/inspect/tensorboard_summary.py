"""Read the latest game-specific scalars without starting TensorBoard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "control/effective_state_hz",
    "combat/current_wave",
    "combat/best_wave",
    "combat/hazard_override_rate",
    "combat/hazard_risk_reduction",
    "combat/damage_taken",
    "combat/death_count_total",
    "movement/reversal_rate",
)


def summarize(logdir: Path, tags: tuple[str, ...] = DEFAULT_TAGS) -> dict:
    files = sorted(
        Path(logdir).rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    runs = []
    latest: dict[str, dict[str, float | int | str]] = {}
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            runs.append({"path": str(path), "error": str(exc)})
            continue
        available = set(accumulator.Tags().get("scalars", []))
        found = []
        for tag in tags:
            if tag not in available:
                continue
            values = accumulator.Scalars(tag)
            if not values:
                continue
            item = values[-1]
            latest[tag] = {
                "value": float(item.value),
                "step": int(item.step),
                "wall_time": float(item.wall_time),
                "event_file": str(path),
            }
            found.append(tag)
        runs.append({"path": str(path), "scalar_tags": len(available), "selected": found})
    return {"logdir": str(Path(logdir).resolve()), "event_files": len(files), "latest": latest, "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = summarize(args.logdir)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
