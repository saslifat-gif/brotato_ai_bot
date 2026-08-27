"""Machine-readable and compact human-readable evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json_report(report: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(report: Mapping[str, Any]) -> str:
    rows = [
        "| Structure | Mean risk | Min-risk action | Override rate | Switches |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["variants"].items():
        rows.append(
            f"| {name} | {metrics['mean_modeled_risk']:.4f} | "
            f"{metrics['minimum_risk_action_rate']:.1%} | "
            f"{metrics['override_rate']:.1%} | {metrics['direction_switches']} |"
        )
    damage = report["observed_outcomes"]
    rows.extend(
        [
            "",
            "Observed outcomes are labels from the fixed recording, not counterfactual claims.",
            "",
            f"- Samples: {report['records']}.",
            f"- Damage samples: {damage['damage_samples']}.",
            f"- Unique 500 ms damage windows: {damage['unique_damage_windows']}.",
            f"- Maximum observed wave: {damage['maximum_wave']}.",
        ]
    )
    return "\n".join(rows) + "\n"


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding="utf-8")

