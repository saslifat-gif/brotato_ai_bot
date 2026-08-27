"""Compare controller structures on the exact same immutable recording."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable

from brotato_ai.control.hazards import UnifiedHazardScorer
from brotato_ai.data.replay import JsonlReplay
from brotato_ai.domain.decisions import HazardRisk
from brotato_ai.evaluation.metrics import DamageMetrics, VariantMetrics
from brotato_ai.evaluation.reports import write_json_report, write_markdown_report


VARIANT_NAMES = (
    "policy_only",
    "projectile_only",
    "enemy_only",
    "unified",
    "unified_stable",
    "noop_analyzer_control",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_value(risk: HazardRisk, component: str) -> float:
    if component == "projectile":
        return risk.projectile_total
    if component == "enemy":
        return risk.enemy_total
    return risk.total


def _select(
    risks: dict[int, HazardRisk],
    requested: int,
    *,
    component: str = "total",
    previous: int | None = None,
    switch_penalty: float = 0.0,
    minimum_risk: float = 0.22,
    hard_threshold: float = 0.65,
    margin: float = 0.08,
) -> int:
    requested_risk = _component_value(risks[requested], component)
    if requested_risk < minimum_risk:
        return requested
    best = min(
        risks,
        key=lambda action: (
            _component_value(risks[action], component)
            + (switch_penalty if previous is not None and action != previous else 0.0),
            action == 0,
            action,
        ),
    )
    best_risk = _component_value(risks[best], component)
    if requested_risk < hard_threshold and requested_risk - best_risk < margin:
        return requested
    return best


def compare_recording(
    recording: Path,
    *,
    max_records: int = 0,
    stride: int = 1,
) -> dict:
    recording = Path(recording)
    scorer = UnifiedHazardScorer(enabled=True, switch_penalty=0.0)
    spacing_off_scorer = UnifiedHazardScorer(
        enabled=True,
        switch_penalty=0.0,
        ranged_spacing_enabled=False,
    )
    metrics = {name: VariantMetrics() for name in VARIANT_NAMES}
    spacing_metrics = {"off": VariantMetrics(), "on": VariantMetrics()}
    damage = DamageMetrics()
    records = 0
    for snapshot, requested in JsonlReplay(
        recording, max_records=max_records, stride=stride
    ).records():
        risks = scorer.all_risks(snapshot)
        spacing_risks_off = spacing_off_scorer.all_risks(snapshot)
        minimum_action = min(risks, key=lambda action: (risks[action].total, action == 0, action))
        spacing_minimums = {
            "off": min(
                spacing_risks_off,
                key=lambda action: (spacing_risks_off[action].total, action == 0, action),
            ),
            "on": minimum_action,
        }
        spacing_risks_by_mode = {"off": spacing_risks_off, "on": risks}
        for mode, mode_risks in spacing_risks_by_mode.items():
            selected_spacing = _select(mode_risks, requested)
            spacing_metrics[mode].observe(
                requested_action=requested,
                selected_action=selected_spacing,
                requested_risk=mode_risks[requested].total,
                selected_risk=mode_risks[selected_spacing].total,
                minimum_action=spacing_minimums[mode],
                unsafe_action_count=sum(
                    risk.total >= 0.65 for risk in mode_risks.values()
                ),
                action_count=len(mode_risks),
                minimum_risk=mode_risks[spacing_minimums[mode]].total,
            )
        selected = {
            "policy_only": requested,
            "projectile_only": _select(risks, requested, component="projectile"),
            "enemy_only": _select(risks, requested, component="enemy"),
            "unified": _select(risks, requested),
            "unified_stable": _select(
                risks,
                requested,
                previous=metrics["unified_stable"].previous_action,
                switch_penalty=0.05,
            ),
            "noop_analyzer_control": requested,
        }
        for name, action in selected.items():
            metrics[name].observe(
                requested_action=requested,
                selected_action=action,
                requested_risk=risks[requested].total,
                selected_risk=risks[action].total,
                minimum_action=minimum_action,
                unsafe_action_count=sum(risk.total >= 0.65 for risk in risks.values()),
                action_count=len(risks),
                minimum_risk=risks[minimum_action].total,
            )
        timestamp_ms = snapshot.timestamp_ms if snapshot.timestamp_ms >= 0 else snapshot.tick * 17
        damage.observe(
            session=snapshot.session,
            timestamp_ms=timestamp_ms,
            health=snapshot.player.health,
            wave=snapshot.wave_number,
            dead=snapshot.dead,
            victory=snapshot.victory,
        )
        records += 1
    variants = {name: value.to_dict() for name, value in metrics.items()}
    drift = abs(
        variants["policy_only"]["mean_modeled_risk"]
        - variants["noop_analyzer_control"]["mean_modeled_risk"]
    )
    return {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": _file_sha256(recording),
        "records": records,
        "stride": max(1, int(stride)),
        "variants": variants,
        "shield_comparison": {
            "off": "policy_only",
            "on": "unified",
            "risk_reduction": (
                variants["policy_only"]["mean_modeled_risk"]
                - variants["unified"]["mean_modeled_risk"]
            ),
            "direction_switch_delta": (
                variants["unified"]["direction_switches"]
                - variants["policy_only"]["direction_switches"]
            ),
        },
        "spacing_comparison": {
            "off": spacing_metrics["off"].to_dict(),
            "on": spacing_metrics["on"].to_dict(),
            "direction_switch_delta": (
                spacing_metrics["on"].direction_switches
                - spacing_metrics["off"].direction_switches
            ),
            "override_rate_delta": (
                spacing_metrics["on"].to_dict()["override_rate"]
                - spacing_metrics["off"].to_dict()["override_rate"]
            ),
        },
        "analyzer_drift": drift,
        "observed_outcomes": damage.to_dict(),
        "interpretation": (
            "Geometric counterfactuals on one fixed recording. Modeled risk does not "
            "prove alternate game outcomes; damage samples and unique windows are separate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = compare_recording(
        args.recording,
        max_records=max(0, args.max_records),
        stride=max(1, args.stride),
    )
    write_json_report(report, args.json)
    write_markdown_report(report, args.markdown)
    print(
        f"[backtest] records={report['records']} drift={report['analyzer_drift']:.8g} "
        f"json={args.json} markdown={args.markdown}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

