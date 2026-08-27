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
        "| Structure | Mean risk | Min-risk action | Unsafe actions | Regret | Override rate | Switches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["variants"].items():
        rows.append(
            f"| {name} | {metrics['mean_modeled_risk']:.4f} | "
            f"{metrics['minimum_risk_action_rate']:.1%} | "
            f"{metrics['mean_unsafe_action_fraction']:.1%} | "
            f"{metrics['mean_requested_to_minimum_regret']:.4f} | "
            f"{metrics['override_rate']:.1%} | {metrics['direction_switches']} |"
        )
    shield = report.get("shield_comparison", {})
    if shield:
        rows.extend(
            [
                "",
                f"- Shield-off structure: {shield.get('off')}; shield-on structure: {shield.get('on')}.",
                f"- Shield modeled-risk delta (off minus on): {shield.get('risk_reduction', 0.0):.4f}.",
                f"- Shield direction-switch delta (on minus off): {shield.get('direction_switch_delta', 0)}.",
            ]
        )
    tactical = report.get("tactical_comparison", {})
    if tactical:
        baseline = tactical.get("baseline_unified", {})
        persistent = tactical.get("persistent_tactical", {})
        rows.extend(
            [
                "",
                "### Tactical escape A/B",
                "",
                f"- Escape entries: baseline {baseline.get('escape_entries', 0)}, persistent {persistent.get('escape_entries', 0)}.",
                f"- Post-escape re-entry rate: baseline {baseline.get('post_escape_reentry_rate', 0.0):.1%}, persistent {persistent.get('post_escape_reentry_rate', 0.0):.1%}.",
                f"- Escape direction reversals: baseline {baseline.get('escape_direction_reversals', 0)}, persistent {persistent.get('escape_direction_reversals', 0)}.",
                f"- Mean enemy separation: baseline {baseline.get('mean_enemy_separation', 0.0):.1f}, persistent {persistent.get('mean_enemy_separation', 0.0):.1f}.",
                "",
                "The tactical comparison is a geometric replay A/B; it does not claim the fixed recording actually followed the counterfactual actions.",
            ]
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

