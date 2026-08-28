"""Classify live deaths with one extra controller call before impact.

This is a geometric replay on a fixed recording. It does not re-simulate
physics. Categories:

- too_late: the last lookback frame already had TTI under 50 ms
- no_safe_action: every action is still high-risk on the impact frame
- wrong_action: a materially safer action existed; the held action was worse
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from brotato_ai.control import CombatDecisionPipeline, CombatSafetyShield, CrowdRecoveryGuard
from brotato_ai.control.hazards import projectile_time_to_impact
from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction

LOOKBACK_MS = 80.0
TOO_LATE_TTI_MS = 50.0
HARD_RISK = 0.65
SAFE_MARGIN = 0.08
COMBAT_PHASES = {"combat", "main"}
DEATH_PHASES = {"game_over", "endrun", "end_run"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _timestamp(raw: Mapping[str, Any]) -> float:
    published = _number(raw.get("published_at_ms"), -1.0)
    if published >= 0.0:
        return published
    recorded = _number(raw.get("recorded_at_ms"), -1.0)
    if recorded >= 0.0:
        return recorded
    return _number(raw.get("tick"), 0.0) * (1000.0 / 60.0)


def _health(raw: Mapping[str, Any]) -> float:
    return _number(_mapping(raw.get("player")).get("health"))


def _wave(raw: Mapping[str, Any]) -> int:
    return int(_number(_mapping(raw.get("wave")).get("number")))


def _action(raw: Mapping[str, Any]) -> int:
    try:
        value = int(raw.get("action", 0))
    except (TypeError, ValueError):
        value = 0
    return value if 0 <= value <= 8 else 0


def _nearest_tti_ms(raw: Mapping[str, Any], action: int) -> float | None:
    player = _mapping(raw.get("player"))
    position = _mapping(player.get("position"))
    combat = _mapping(raw.get("combat"))
    speed = max(150.0, _number(combat.get("move_speed"), 300.0))
    projectiles = raw.get("projectiles")
    if not isinstance(projectiles, Iterable) or isinstance(projectiles, (str, bytes, Mapping)):
        return None
    values: list[float] = []
    movement = ACTION_VECTORS[MoveAction(int(action))]
    origin = (_number(position.get("x")), _number(position.get("y")))
    for projectile in projectiles:
        if not isinstance(projectile, Mapping):
            continue
        if projectile.get("hostile", True) is False:
            continue
        tti, miss = projectile_time_to_impact(projectile, origin, movement, speed)
        radius = max(8.0, _number(projectile.get("radius"), 12.0)) + 42.0
        if miss <= radius:
            values.append(tti * 1000.0)
    return min(values) if values else None


def _pipeline() -> CombatDecisionPipeline:
    shield = CombatSafetyShield(enabled=True)
    return CombatDecisionPipeline(
        safety_shield=shield,
        crowd_recovery_guard=CrowdRecoveryGuard(enabled=True, shield=shield),
    )


def _classify_one(
    lookback: Mapping[str, Any],
    impact: Mapping[str, Any],
    *,
    pipeline: CombatDecisionPipeline,
) -> dict[str, Any]:
    recorded = _action(lookback)
    extra = int(
        pipeline.apply(
            lookback,
            recorded,
            previous_action=recorded,
            control_interval_ms=LOOKBACK_MS,
        ).decision.applied_action
    )
    risks = pipeline.safety_shield.all_risks(impact)
    recorded_risk = float(risks[recorded].total)
    extra_risk = float(risks[extra].total)
    safest = min(risks, key=lambda action: (risks[action].total, action == 0, action))
    safest_risk = float(risks[safest].total)
    tti = _nearest_tti_ms(lookback, recorded)
    if tti is not None and tti < TOO_LATE_TTI_MS:
        category = "too_late"
    elif safest_risk >= HARD_RISK:
        category = "no_safe_action"
    elif recorded_risk - safest_risk >= SAFE_MARGIN:
        category = "wrong_action"
    else:
        category = "already_best"
    return {
        "category": category,
        "wave": _wave(impact),
        "lookback_ms": _timestamp(impact) - _timestamp(lookback),
        "health_before": _health(lookback),
        "health_after": _health(impact),
        "dead": bool(impact.get("dead")),
        "recorded_action": recorded,
        "extra_action": extra,
        "safest_action": int(safest),
        "recorded_risk": recorded_risk,
        "extra_risk": extra_risk,
        "safest_risk": safest_risk,
        "lookback_tti_ms": tti,
        "enemies": len(impact.get("enemies") or []),
        "projectiles": len(impact.get("projectiles") or []),
        "extra_helped": extra_risk + SAFE_MARGIN < recorded_risk and extra_risk < HARD_RISK,
    }


def _lookback_frame(
    window: deque[Mapping[str, Any]], impact_ts: float
) -> Mapping[str, Any] | None:
    target = impact_ts - LOOKBACK_MS
    chosen: Mapping[str, Any] | None = None
    for raw in window:
        if _timestamp(raw) <= target:
            chosen = raw
        else:
            break
    if chosen is None and window:
        chosen = window[0]
    if chosen is not None and abs(_timestamp(chosen) - impact_ts) < 5.0:
        return None
    return chosen


def classify_recording(path: Path) -> dict[str, Any]:
    pipeline = _pipeline()
    window: deque[Mapping[str, Any]] = deque()
    deaths: list[dict[str, Any]] = []
    last_health: float | None = None
    last_dead = False
    frames = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(raw, dict) or raw.get("type") not in {"raw_state", "state"}:
                continue
            phase = str(raw.get("phase", "")).lower()
            if phase in DEATH_PHASES:
                if window and not last_dead:
                    impact = window[-1]
                    lookback = _lookback_frame(window, _timestamp(impact))
                    if lookback is not None:
                        event = _classify_one(lookback, impact, pipeline=pipeline)
                        event["kind"] = "death"
                        event["session"] = str(impact.get("session", ""))
                        event["death_phase"] = str(raw.get("phase"))
                        deaths.append(event)
                last_health = None
                last_dead = True
                window.clear()
                continue
            if phase not in COMBAT_PHASES:
                last_health = None
                last_dead = bool(raw.get("dead"))
                window.clear()
                continue
            frames += 1
            health = _health(raw)
            dead = bool(raw.get("dead")) or health <= 0.0
            now = _timestamp(raw)
            window.append(raw)
            while window and now - _timestamp(window[0]) > 2000.0:
                window.popleft()
            death_edge = dead and not last_dead
            if death_edge:
                lookback = _lookback_frame(window, now)
                if lookback is not None:
                    event = _classify_one(lookback, raw, pipeline=pipeline)
                    event["kind"] = "death"
                    event["session"] = str(raw.get("session", ""))
                    deaths.append(event)
            last_health = health
            last_dead = dead
    counts: dict[str, int] = {}
    for event in deaths:
        counts[event["category"]] = counts.get(event["category"], 0) + 1
    extra_helped = sum(int(bool(event.get("extra_helped"))) for event in deaths)
    return {
        "recording": str(path),
        "combat_frames": frames,
        "deaths_classified": len(deaths),
        "categories": counts,
        "extra_decision_would_have_helped": extra_helped,
        "events": deaths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify live recording deaths")
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    reports = [classify_recording(path) for path in args.recordings]
    totals: dict[str, int] = {}
    deaths = 0
    helped = 0
    for report in reports:
        deaths += int(report["deaths_classified"])
        helped += int(report["extra_decision_would_have_helped"])
        for key, value in report["categories"].items():
            totals[key] = totals.get(key, 0) + int(value)
    summary = {
        "lookback_ms": LOOKBACK_MS,
        "too_late_tti_ms": TOO_LATE_TTI_MS,
        "hard_risk": HARD_RISK,
        "deaths_classified": deaths,
        "categories": totals,
        "extra_decision_would_have_helped": helped,
        "recordings": [
            {
                "recording": Path(report["recording"]).name,
                "deaths_classified": report["deaths_classified"],
                "categories": report["categories"],
                "extra_decision_would_have_helped": report["extra_decision_would_have_helped"],
            }
            for report in reports
        ],
        "events": [event for report in reports for event in report["events"]],
        "interpretation": (
            "Geometric replay on live recordings. too_late = TTI < 50 ms at the "
            "lookback frame. no_safe_action = all nine actions still high-risk at "
            "impact. wrong_action = a safer lane existed. already_best = held "
            "action was already the safest scored lane."
        ),
    }
    text = json.dumps(summary, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(
        f"deaths={deaths} too_late={totals.get('too_late', 0)} "
        f"no_safe={totals.get('no_safe_action', 0)} "
        f"wrong={totals.get('wrong_action', 0)} "
        f"already_best={totals.get('already_best', 0)} "
        f"extra_helped={helped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
