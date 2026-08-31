"""Generate a human-readable and structured summary of shadow/frozen evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ACTION_NAMES = (
    "IDLE", "UP", "DOWN", "LEFT", "RIGHT",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
)


def format_shadow_report(results: list[Mapping[str, Any]]) -> str:
    if not results:
        return "No evaluation episodes found."

    episodes = len(results)
    rewards = [float(r.get("reward", 0.0)) for r in results]
    steps = [int(r.get("steps", 0)) for r in results]
    waves = [int(r.get("wave", 0)) for r in results]
    shield_overrides = [int(r.get("shield_overrides", 0)) for r in results]

    mean_reward = sum(rewards) / max(1, episodes)
    max_reward = max(rewards) if rewards else 0.0
    min_reward = min(rewards) if rewards else 0.0

    total_steps = sum(steps)
    mean_steps = total_steps / max(1, episodes)

    mean_wave = sum(waves) / max(1, episodes)
    max_wave = max(waves) if waves else 0

    total_shield = sum(shield_overrides)
    shield_rate = (total_shield / max(1, total_steps)) * 100.0

    # Aggregate action counts
    action_totals = [0] * len(ACTION_NAMES)
    for r in results:
        counts = r.get("requested_action_counts", [])
        for i, count in enumerate(counts[:len(action_totals)]):
            action_totals[i] += int(count)

    lines = [
        "# Brotato AI Policy Evaluation Summary",
        "",
        "## Overall Metrics",
        "",
        f"- **Episodes Evaluated**: {episodes}",
        f"- **Mean Reward**: {mean_reward:.3f} (min: {min_reward:.3f}, max: {max_reward:.3f})",
        f"- **Mean Wave Reached**: {mean_wave:.1f} (max: {max_wave})",
        f"- **Total Combat Steps**: {total_steps} (mean: {mean_steps:.1f})",
        f"- **Safety Shield Overrides**: {total_shield} ({shield_rate:.2f}% of steps)",
        "",
        "## Episode Breakdown",
        "",
        "| Episode | Policy | Wave | Steps | Reward | Shield Overrides | Shield Rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in results:
        ep_num = r.get("episode", "?")
        pol = r.get("policy", "unknown")
        wv = r.get("wave", 0)
        st = r.get("steps", 0)
        rew = float(r.get("reward", 0.0))
        sh = int(r.get("shield_overrides", 0))
        sh_r = (sh / max(1, st)) * 100.0
        lines.append(f"| {ep_num} | {pol} | {wv} | {st} | {rew:.3f} | {sh} | {sh_r:.2f}% |")

    lines.extend([
        "",
        "## Action Distribution",
        "",
        "| Action | Count | Percentage |",
        "| --- | ---: | ---: |",
    ])

    for action_idx, action_name in enumerate(ACTION_NAMES):
        count = action_totals[action_idx]
        pct = (count / max(1, total_steps)) * 100.0
        lines.append(f"| {action_name} | {count} | {pct:.1f}% |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report and summarize shadow/frozen evaluation runs")
    parser.add_argument("results_file", type=Path, help="Path to evaluation JSON results file (e.g. reports/shadow.json)")
    parser.add_argument("--output", "-o", type=Path, help="Optional output path for Markdown summary")
    args = parser.parse_args(argv)

    if not args.results_file.is_file():
        print(f"[v4-report-shadow] Error: file not found: {args.results_file}", file=sys.stderr)
        return 1

    try:
        data = json.loads(args.results_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[v4-report-shadow] Error parsing {args.results_file}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        data = [data]

    report = format_shadow_report(data)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"[v4-report-shadow] Saved report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
