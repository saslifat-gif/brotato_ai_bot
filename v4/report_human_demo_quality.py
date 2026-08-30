"""Create a concise quality report for a set of manual human demonstrations.

This is an offline/research report. It reads recorder outputs only; it does not
connect to Brotato, send actions, train a model, or change policy configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from brotato_ai.data.human_demo import _from_blob, validate_dataset


ACTION_NAMES = (
    "IDLE", "UP", "DOWN", "LEFT", "RIGHT",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _entropy(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    return -sum((value / total) * math.log2(value / total) for value in counts.values())


def _weapon_ids(state: Mapping[str, Any]) -> list[str]:
    combat = _mapping(state.get("combat"))
    build = _mapping(state.get("build"))
    weapons = _items(combat.get("weapons")) or _items(build.get("weapons"))
    identifiers: list[str] = []
    for item in weapons:
        value = item.get("id") or item.get("my_id") or item.get("weapon_id") or item.get("type")
        if value not in (None, ""):
            identifiers.append(str(value))
    return sorted(set(identifiers))


def _build_signature(state: Mapping[str, Any]) -> str:
    identifiers = _weapon_ids(state)
    if identifiers:
        return "+".join(identifiers)
    combat = _mapping(state.get("combat"))
    count = int(_number(combat.get("weapon_count"), 0))
    return f"weapon_count_{count}" if count else "unknown"


def _build_class(signature: str) -> str:
    lowered = signature.lower()
    if any(token in lowered for token in ("smg", "submachine")):
        return "SMG"
    if signature == "unknown" or signature.startswith("weapon_count_"):
        return "unknown"
    return "non-SMG"


def _frame_rows(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(str(path))
    rows = connection.execute(
        """
        SELECT frame_id,episode_id,frame_number,timestamp_ns,phase,wave,action,
               previous_action,state_blob,derived_blob,controller_blob,feature_blob
        FROM frames ORDER BY episode_id,frame_number,frame_id
        """
    ).fetchall()
    connection.close()
    output = []
    for row in rows:
        output.append({
            "frame_id": int(row[0]),
            "episode_id": str(row[1]),
            "frame_number": int(row[2]),
            "timestamp_ns": int(row[3]),
            "phase": str(row[4]),
            "wave": int(row[5] or 0),
            "action": int(row[6]),
            "previous_action": int(row[7]),
            "state": _from_blob(row[8], {}),
            "derived": _from_blob(row[9], {}),
            "controller": _from_blob(row[10], {}),
            "features": _from_blob(row[11], []),
        })
    return output


def _frame_context(row: Mapping[str, Any]) -> dict[str, Any]:
    state = _mapping(row.get("state"))
    derived = _mapping(row.get("derived"))
    player = _mapping(state.get("player"))
    hp = _number(player.get("health"), 0.0) / max(1.0, _number(player.get("max_health"), 1.0))
    enemy_count = int(_number(derived.get("enemy_count"), len(_items(state.get("enemies")))))
    projectile_count = int(_number(derived.get("projectile_count"), len(_items(state.get("projectiles")))))
    indicators = int(_number(derived.get("telegraph_count"), len(_items(state.get("attack_indicators")))))
    hazard = bool(derived.get("hazard_actionable"))
    recovery = bool(
        derived.get("escape")
        or derived.get("recovery_active")
        or _mapping(row.get("controller")).get("source") in {"hazard", "crowd_recovery"}
    )
    return {
        "wave": int(row.get("wave") or _number(_mapping(state.get("wave")).get("number"), 0)),
        "hp_fraction": hp,
        "enemy_count": enemy_count,
        "projectile_count": projectile_count,
        "indicator_count": indicators,
        "hazard": hazard,
        "recovery": recovery,
        "build": _build_signature(state),
        "build_class": _build_class(_build_signature(state)),
    }


def _parity_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare recorded features with the training/live event input contract."""

    try:
        import numpy as np
        from brotato_ai.policy.features import HumanPolicyFeatureBuilder, zero_previous_action_slice
    except Exception as exc:  # pragma: no cover - environment-dependent dependency
        return {"status": "unavailable", "reason": f"feature dependencies unavailable: {exc}"}

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    checked = 0
    state_feature_failures = 0
    input_failures = 0
    max_state_error = 0.0
    max_input_error = 0.0
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: (row["frame_number"], row["frame_id"]))
        builder = HumanPolicyFeatureBuilder()
        timestamps = np.maximum.accumulate(
            np.asarray([row["timestamp_ns"] for row in episode_rows], dtype=np.int64)
        )
        stored_states = []
        for index, row in enumerate(episode_rows):
            features = np.asarray(row["features"], dtype=np.float32)
            if features.size != 832:
                continue
            observed = builder.observe(
                _mapping(row["state"]),
                int(row["previous_action"]),
                timestamp_ms=float(timestamps[index]) / 1e6,
            )
            expected_state = zero_previous_action_slice(features)
            state_error = float(np.max(np.abs(observed - expected_state)))
            max_state_error = max(max_state_error, state_error)
            if state_error > 1e-5:
                state_feature_failures += 1
            stored_states.append(features)
            live_input = builder.build_input(int(row["previous_action"]))
            snapshots = []
            for offset_ms in (0.0, 200.0, 400.0):
                target = int(timestamps[index] - offset_ms * 1e6)
                history_index = int(np.searchsorted(timestamps[: index + 1], target, side="right") - 1)
                history_index = min(index, max(0, history_index))
                snapshots.append(zero_previous_action_slice(
                    np.asarray(episode_rows[history_index]["features"], dtype=np.float32)
                ))
            action_one_hot = np.zeros(9, dtype=np.float32)
            previous_action = int(row["previous_action"])
            if 0 <= previous_action < 9:
                action_one_hot[previous_action] = 1.0
            expected_input = np.concatenate(
                (snapshots[0], snapshots[0] - snapshots[1], snapshots[0] - snapshots[2], action_one_hot)
            ).astype(np.float32)
            input_error = float(np.max(np.abs(live_input - expected_input)))
            max_input_error = max(max_input_error, input_error)
            if input_error > 1e-5:
                input_failures += 1
            checked += 1
    return {
        "status": "pass" if not state_feature_failures and not input_failures else "fail",
        "frames_checked": checked,
        "state_feature_failures": state_feature_failures,
        "event_input_failures": input_failures,
        "max_state_error": max_state_error,
        "max_event_input_error": max_input_error,
        "definition": "recorded SemanticCombatVectorizer features and 200/400 ms event history compared with HumanPolicyFeatureBuilder",
    }


def _run_report(path: Path) -> dict[str, Any]:
    validation = validate_dataset(path)
    rows = _frame_rows(path)
    combat = [row for row in rows if row["phase"] == "combat"]
    transitions = [
        row for row in combat
        if row["frame_number"] > 0 and row["action"] != row["previous_action"]
    ]
    transition_pairs = Counter(
        f"{ACTION_NAMES[row['previous_action']]}->{ACTION_NAMES[row['action']]}"
        for row in transitions
        if 0 <= row["previous_action"] < 9 and 0 <= row["action"] < 9
    )
    action_counts = Counter(
        ACTION_NAMES[row["action"]] for row in combat if 0 <= row["action"] < 9
    )
    build_frames = Counter(_frame_context(row)["build"] for row in combat)
    build_classes = Counter(_frame_context(row)["build_class"] for row in combat)
    situations = {
        "waves_8_12": sum(8 <= _frame_context(row)["wave"] <= 12 for row in combat),
        "late_wave_8_plus": sum(_frame_context(row)["wave"] >= 8 for row in combat),
        "low_hp_below_50pct": sum(_frame_context(row)["hp_fraction"] < 0.5 for row in combat),
        "very_low_hp_below_25pct": sum(_frame_context(row)["hp_fraction"] < 0.25 for row in combat),
        "dense_enemies_10_plus": sum(_frame_context(row)["enemy_count"] >= 10 for row in combat),
        "dense_projectiles_5_plus": sum(_frame_context(row)["projectile_count"] >= 5 for row in combat),
        "visible_telegraphs": sum(_frame_context(row)["indicator_count"] > 0 for row in combat),
        "hazard_actionable": sum(_frame_context(row)["hazard"] for row in combat),
        "recovery_frames": sum(_frame_context(row)["recovery"] for row in combat),
        "non_smg_frames": sum(_frame_context(row)["build_class"] == "non-SMG" for row in combat),
    }
    connection = sqlite3.connect(str(path))
    segment_durations = [
        float(value) for (value,) in connection.execute(
            "SELECT duration_ms FROM action_segments WHERE duration_ms IS NOT NULL AND duration_ms >= 0"
        ).fetchall()
    ]
    build_rows = connection.execute(
        "SELECT available_blob,build_before_blob,build_after_blob,chosen_action,source FROM build_decisions"
    ).fetchall()
    episodes = connection.execute(
        "SELECT episode_id,outcome FROM episodes ORDER BY started_ns,episode_id"
    ).fetchall()
    connection.close()
    available_choice_snapshots = 0
    available_options = 0
    selected_build_decisions = 0
    for available_blob, _before, _after, chosen, _source in build_rows:
        available = _from_blob(available_blob, [])
        if isinstance(available, list):
            available_choice_snapshots += 1
            available_options += len(available)
        selected_build_decisions += int(chosen is not None)
    wave_values = [_frame_context(row)["wave"] for row in rows]
    return {
        "path": str(path.resolve()),
        "validation": validation,
        "episodes": len(episodes),
        "outcomes": Counter(str(outcome or "unknown") for _episode, outcome in episodes),
        "combat_frames": len(combat),
        "genuine_action_transitions": len(transitions),
        "action_counts": dict(action_counts),
        "transition_pairs": dict(transition_pairs),
        "transition_diversity": {
            "unique_actions": len(action_counts),
            "unique_transition_pairs": len(transition_pairs),
            "transition_entropy_bits": _entropy(transition_pairs),
            "median_hold_ms": sorted(segment_durations)[len(segment_durations) // 2] if segment_durations else None,
            "p90_hold_ms": sorted(segment_durations)[min(len(segment_durations) - 1, int(len(segment_durations) * 0.9))] if segment_durations else None,
        },
        "build_decisions": {
            "snapshots": len(build_rows),
            "snapshots_with_available_choices": available_choice_snapshots,
            "available_choice_count": available_options,
            "selected_or_inferred_count": selected_build_decisions,
        },
        "build_coverage": {
            "signatures": dict(build_frames),
            "classes": dict(build_classes),
        },
        "situations": situations,
        "wave_max": max(wave_values, default=0),
        "feature_parity": _parity_check(rows),
    }


def _aggregate(run_reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "runs": len(run_reports),
        "episodes": sum(report["episodes"] for report in run_reports),
        "combat_frames": sum(report["combat_frames"] for report in run_reports),
        "genuine_action_transitions": sum(report["genuine_action_transitions"] for report in run_reports),
        "build_decisions": sum(report["build_decisions"]["snapshots"] for report in run_reports),
        "waves_reached_max": max((report["wave_max"] for report in run_reports), default=0),
        "deaths": sum(int(report["outcomes"].get("death", 0)) for report in run_reports),
        "victories": sum(int(report["outcomes"].get("victory", 0)) for report in run_reports),
    }
    situation_totals = Counter()
    build_signatures = Counter()
    build_classes = Counter()
    transition_pairs = Counter()
    for report in run_reports:
        situation_totals.update(report["situations"])
        build_signatures.update(report["build_coverage"]["signatures"])
        build_classes.update(report["build_coverage"]["classes"])
        transition_pairs.update(report["transition_pairs"])
    combat_frames = max(1, totals["combat_frames"])
    situation_coverage = {
        key: {"frames": int(value), "fraction": float(value / combat_frames)}
        for key, value in sorted(situation_totals.items())
    }
    underrepresented = [
        key for key, value in situation_coverage.items()
        if value["frames"] == 0 or value["fraction"] < 0.01
    ]
    parity_statuses = [report["feature_parity"].get("status") for report in run_reports]
    validation_ready = all(bool(report["validation"].get("capture_ready")) for report in run_reports)
    parity_ready = bool(parity_statuses) and all(status == "pass" for status in parity_statuses)
    minimums = {
        "three_run_files": totals["runs"] >= 3,
        "combat_frames": totals["combat_frames"] >= 1000,
        "genuine_transitions": totals["genuine_action_transitions"] >= 100,
        "late_waves_8_12": situation_totals["waves_8_12"] >= 100,
        "non_smg_coverage": situation_totals["non_smg_frames"] >= 50,
        "low_hp_coverage": situation_totals["low_hp_below_50pct"] >= 10,
        "dense_enemy_coverage": situation_totals["dense_enemies_10_plus"] >= 10,
        "dense_projectile_coverage": situation_totals["dense_projectiles_5_plus"] >= 10,
        "recovery_coverage": situation_totals["recovery_frames"] >= 10,
    }
    suitable = validation_ready and parity_ready and all(minimums.values())
    reasons = []
    if not validation_ready:
        reasons.append("one or more run files failed complete-capture validation")
    if not parity_ready:
        reasons.append("training/live event-feature parity did not pass for every run")
    reasons.extend(f"insufficient {key.replace('_', ' ')}" for key, value in minimums.items() if not value)
    return {
        "totals": totals,
        "situation_coverage": situation_coverage,
        "underrepresented_situations": underrepresented,
        "build_coverage": {
            "signatures": dict(build_signatures),
            "classes": dict(build_classes),
        },
        "transition_diversity": {
            "unique_transition_pairs": len(transition_pairs),
            "transition_entropy_bits": _entropy(transition_pairs),
            "pairs": dict(transition_pairs),
        },
        "feature_parity_statuses": parity_statuses,
        "minimum_checks": minimums,
        "suitable_for_event_policy_retraining": suitable,
        "suitability_reasons": reasons or ["all capture, parity, and coverage checks passed"],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    totals = aggregate["totals"]
    lines = [
        "# Human demonstration dataset quality",
        "",
        f"Decision: **{'SUITABLE' if aggregate['suitable_for_event_policy_retraining'] else 'NOT YET SUITABLE'}** for event-policy retraining.",
        "",
        f"Runs={totals['runs']} episodes={totals['episodes']} combat_frames={totals['combat_frames']} "
        f"genuine_transitions={totals['genuine_action_transitions']} build_decisions={totals['build_decisions']} "
        f"max_wave={totals['waves_reached_max']} deaths={totals['deaths']} victories={totals['victories']}",
        "",
        "## Coverage",
        "",
        "| Situation | Frames | Fraction |\n|---|---:|---:|",
    ]
    for key, value in aggregate["situation_coverage"].items():
        lines.append(f"| {key} | {value['frames']} | {value['fraction']:.2%} |")
    lines.extend([
        "",
        f"Build classes: `{json.dumps(aggregate['build_coverage']['classes'], sort_keys=True)}`",
        f"Underrepresented: `{', '.join(aggregate['underrepresented_situations']) or 'none'}`",
        "",
        "## Parity and checks",
        "",
        f"Feature parity statuses: `{aggregate['feature_parity_statuses']}`",
        f"Checks: `{json.dumps(aggregate['minimum_checks'], sort_keys=True)}`",
        f"Reasons: {'; '.join(aggregate['suitability_reasons'])}",
        "",
        "The report is observational; no controller mode, policy checkpoint, or live action was changed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report quality of manual human-demo SQLite files")
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--markdown", type=Path, help="optional Markdown report path")
    args = parser.parse_args()
    missing = [str(path) for path in args.datasets if not path.is_file()]
    if missing:
        raise SystemExit(f"dataset files not found: {', '.join(missing)}")
    runs = [_run_report(path) for path in args.datasets]
    report = {
        "schema": 1,
        "purpose": "manual human-demonstration capture validation",
        "runs": runs,
        "aggregate": _aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0 if report["aggregate"]["suitable_for_event_policy_retraining"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
