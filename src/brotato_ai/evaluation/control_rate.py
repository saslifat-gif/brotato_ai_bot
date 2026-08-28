"""Control-rate sensitivity experiment for fixed 60 Hz state recordings.

This module deliberately changes only the schedule at which the existing v4
decision pipeline is called.  Recorded observations are never queued: at each
scheduled tick the newest observation whose timestamp has arrived is used and
the resulting action is held until the next scheduled tick.

The game state in a passive recording is not re-simulated.  Consequently,
health/death labels are reported as observed labels, while action-dependent
quantities are explicitly reported as modeled replay quantities.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from brotato_ai.control import CombatDecisionPipeline, CombatSafetyShield, CrowdRecoveryGuard
from brotato_ai.control.hazards import projectile_time_to_impact
from brotato_ai.data.replay import JsonlReplay
from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction
from brotato_ai.domain.state import StateSnapshot


# Keep the lower references from the original sweep, and include the exact
# live-comparison rates requested by the bridge investigation.
RATES_HZ = (10.0, 15.0, 20.0, 24.0, 30.0, 40.0, 60.0)
PHASE_OFFSETS_MS = (0.0, 16.667, 33.333, 50.0)
TTI_BUCKETS = ((0.0, 50.0, "<50ms"), (50.0, 100.0, "50-100ms"),
               (100.0, 150.0, "100-150ms"), (150.0, 250.0, "150-250ms"),
               (250.0, 400.0, "250-400ms"), (400.0, float("inf"), ">400ms"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload(snapshot: StateSnapshot) -> Mapping[str, Any]:
    return snapshot.payload


def _timestamp(snapshot: StateSnapshot) -> float:
    if snapshot.timestamp_ms >= 0:
        return float(snapshot.timestamp_ms)
    return float(snapshot.tick) * (1000.0 / 60.0)


def _projectile_tti(snapshot: StateSnapshot | Mapping[str, Any], action: int) -> float | None:
    payload = _payload(snapshot) if isinstance(snapshot, StateSnapshot) else snapshot
    player = payload.get("player", {})
    pos = player.get("position", {}) if isinstance(player, Mapping) else {}
    combat = payload.get("combat", {})
    speed = float(combat.get("move_speed", 300.0)) if isinstance(combat, Mapping) else 300.0
    values: list[float] = []
    projectiles = payload.get("projectiles", [])
    if not isinstance(projectiles, Iterable) or isinstance(projectiles, (str, bytes, Mapping)):
        return None
    for projectile in projectiles:
        if not isinstance(projectile, Mapping) or projectile.get("hostile", True) is False:
            continue
        tti, miss = projectile_time_to_impact(
            projectile, (float(pos.get("x", 0.0)), float(pos.get("y", 0.0))),
            ACTION_VECTORS[MoveAction(int(action))], speed
        )
        radius = max(8.0, float(projectile.get("radius", 12.0))) + 42.0
        if miss <= radius:
            values.append(tti * 1000.0)
    return min(values) if values else None


def _health(snapshot: StateSnapshot) -> float:
    return float(snapshot.player.health)


def _sessions(records: list[tuple[StateSnapshot, int]]) -> dict[str, list[tuple[StateSnapshot, int]]]:
    grouped: dict[str, list[tuple[StateSnapshot, int]]] = {}
    for row in records:
        grouped.setdefault(row[0].session, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: _timestamp(row[0]))
    return grouped


@dataclass
class ReplayResult:
    rate_hz: float
    phase_offset_ms: float
    decisions: int
    frames: int
    health_loss_events: int
    death_events: int
    hazard_windows: int
    failed_hazard_windows: int
    projectile_hits: int
    escape_entries: int
    escape_reversals: int
    escape_time_ms: float
    ranged_spacing_time_ms: float
    total_time_ms: float
    safest_actions: int
    no_safe_by_next_tick: int
    actionable_delays_ms: list[float]
    tti_failures_ms: list[float]
    failure_categories: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        delays = sorted(self.actionable_delays_ms)
        def percentile(p: float) -> float | None:
            if not delays:
                return None
            index = min(len(delays) - 1, max(0, math.ceil(p * len(delays)) - 1))
            return delays[index]
        return {
            "rate_hz": self.rate_hz,
            "phase_offset_ms": self.phase_offset_ms,
            "decisions": self.decisions,
            "frames": self.frames,
            "health_loss_rate": self.health_loss_events / max(1, self.frames),
            "health_loss_events_observed": self.health_loss_events,
            "death_collision_rate_observed": self.death_events / max(1, getattr(self, "sessions", 0)),
            "death_events_observed": self.death_events,
            "hazard_window_failure_rate": self.failed_hazard_windows / max(1, self.hazard_windows),
            "hazard_windows": self.hazard_windows,
            "projectile_hit_rate_observed": self.projectile_hits / max(1, self.hazard_windows),
            "projectile_hits_observed": self.projectile_hits,
            "post_escape_reentry_rate": self._reentry_rate,
            "escape_entries": self.escape_entries,
            "escape_direction_reversals": self.escape_reversals,
            "mean_enemy_separation": self._mean_separation,
            "minimum_enemy_separation": self._minimum_separation,
            "safest_action_selection_rate": self.safest_actions / max(1, self.decisions),
            "action_changes": getattr(self, "action_changes", 0),
            "action_change_frequency_hz": getattr(self, "action_changes", 0)
            / max(0.001, self.total_time_ms / 1000.0),
            "action_oscillations": getattr(self, "action_oscillations", 0),
            "action_oscillation_rate_hz": getattr(self, "action_oscillations", 0)
            / max(0.001, self.total_time_ms / 1000.0),
            "action_hold_mean_ms": (
                mean(getattr(self, "action_hold_ms", []))
                if getattr(self, "action_hold_ms", [])
                else None
            ),
            "action_hold_max_ms": (
                max(getattr(self, "action_hold_ms", []))
                if getattr(self, "action_hold_ms", [])
                else None
            ),
            "mean_stale_observation_ms_at_decision": (
                mean(getattr(self, "stale_observation_ms", []))
                if getattr(self, "stale_observation_ms", [])
                else None
            ),
            "maximum_stale_observation_ms_at_decision": (
                max(getattr(self, "stale_observation_ms", []))
                if getattr(self, "stale_observation_ms", [])
                else None
            ),
            "step_based_persistence": {
                "crowd_recovery_hold_steps": 8,
                "crowd_recovery_effective_hold_ms": 8 * 1000.0 / self.rate_hz,
                "projectile_override_hold_steps": 1,
                "projectile_override_effective_hold_ms": 1000.0 / self.rate_hz,
                "note": "These are unchanged controller step counts; their wall-clock duration necessarily varies with the tested decision rate.",
            },
            "time_in_escape_fraction": self.escape_time_ms / max(1.0, self.total_time_ms),
            "desired_ranged_spacing_fraction": self.ranged_spacing_time_ms / max(1.0, self.total_time_ms),
            "no_safe_action_by_next_control_tick": self.no_safe_by_next_tick,
            "actionable_to_next_tick_ms": {
                "mean": mean(delays) if delays else None,
                "median": median(delays) if delays else None,
                "p90": percentile(.90), "p95": percentile(.95),
                "maximum": max(delays) if delays else None,
            },
            "failed_hazard_tti_buckets": _bucket_counts(self.tti_failures_ms),
            "counterfactual_failure_categories": self.failure_categories,
            "counterfactual_extra_decision": getattr(
                self, "counterfactual_extra_decision", {}
            ),
            "interpretation": "Health/death/hit labels are observed on the fixed recording. Action-dependent metrics are geometric/model replay measurements, not a claim that the recorded game state would have changed.",
        }

    @property
    def _reentry_rate(self) -> float:
        return float(getattr(self, "reentries", 0)) / max(1, self.escape_entries)

    @property
    def _mean_separation(self) -> float | None:
        values = getattr(self, "separations", [])
        return mean(values) if values else None

    @property
    def _minimum_separation(self) -> float | None:
        values = getattr(self, "separations", [])
        return min(values) if values else None


def _bucket_counts(values: list[float]) -> dict[str, int]:
    return {label: sum(lo <= value < hi for value in values) for lo, hi, label in TTI_BUCKETS}


def _next_tick_at_or_after(now: float, scheduled: float, interval: float) -> float:
    if scheduled >= now:
        return scheduled
    return scheduled + max(1, math.ceil((now - scheduled) / interval)) * interval


def _enemy_separation(snapshot: StateSnapshot) -> float | None:
    px, py = snapshot.player.position.x, snapshot.player.position.y
    distances = [math.hypot(enemy.position.x - px, enemy.position.y - py) for enemy in snapshot.enemies]
    return min(distances) if distances else None


def _in_desired_spacing(snapshot: StateSnapshot) -> bool:
    payload = _payload(snapshot)
    combat = payload.get("combat", {})
    preferred = float(combat.get("weapon_range", 170.0)) if isinstance(combat, Mapping) else 170.0
    separation = _enemy_separation(snapshot)
    return separation is not None and preferred * 0.75 <= separation <= preferred * 1.75


def _actionable(
    state: Mapping[str, Any],
    scorer: CombatSafetyShield,
    baseline_action: int | None = None,
) -> tuple[bool, float | None, float]:
    risks = scorer.all_risks(state)
    baseline = int(MoveAction(int(baseline_action))) if baseline_action is not None else int(MoveAction.IDLE)
    current = risks[baseline].total
    best_action = min(
        risks,
        key=lambda action: (risks[action].total, action == int(MoveAction.IDLE), action),
    )
    safe_alternative = (
        best_action != baseline
        and current >= scorer.minimum_risk
        and current - risks[best_action].total >= scorer.override_margin
    )
    tti = _projectile_tti(state, baseline)
    return bool(safe_alternative), tti, risks[best_action].total


class _ReplayCachedShield(CombatSafetyShield):
    """Reuse the exact per-state risk scan across rate conditions."""

    def __init__(self, diagnostics: Mapping[int, tuple[bool, float | None, dict[int, Any]]]):
        super().__init__(enabled=True, switch_penalty=0.05)
        self._diagnostics = diagnostics

    @staticmethod
    def _key(state: StateSnapshot | Mapping[str, Any]) -> int:
        return id(state.payload) if isinstance(state, StateSnapshot) else id(state)

    def all_risks(self, state: StateSnapshot | Mapping[str, Any]) -> dict[int, Any]:
        cached = self._diagnostics.get(self._key(state))
        return cached[2] if cached is not None else super().all_risks(state)

    def risk(self, state: StateSnapshot | Mapping[str, Any], action: int) -> float:
        return self.all_risks(state)[int(action)].total


def _new_pipeline(
    diagnostics: Mapping[int, tuple[bool, float | None, dict[int, Any]]] | None = None,
) -> CombatDecisionPipeline:
    scorer = (
        _ReplayCachedShield(diagnostics)
        if diagnostics is not None
        else CombatSafetyShield(enabled=True, switch_penalty=0.05)
    )
    return CombatDecisionPipeline(safety_shield=scorer, crowd_recovery_guard=CrowdRecoveryGuard(enabled=True, shield=scorer))


def run_rate(records: list[tuple[StateSnapshot, int]], rate_hz: float, phase_offset_ms: float = 0.0, diagnostics: Mapping[int, tuple[bool, float | None, dict[int, Any]]] | None = None) -> ReplayResult:
    grouped = _sessions(records)
    scorer = CombatSafetyShield(enabled=True, switch_penalty=0.05)
    result = ReplayResult(rate_hz, phase_offset_ms, 0, len(records), 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, [], [], {})
    result.separations = []
    result.reentries = 0
    result.sessions = len(grouped)
    result.action_changes = 0
    result.action_oscillations = 0
    result.action_hold_ms = []
    result.stale_observation_ms = []
    result.failure_events = []
    interval = 1000.0 / float(rate_hz)
    for rows in grouped.values():
        if not rows:
            continue
        pipeline = _new_pipeline(diagnostics)
        first = _timestamp(rows[0][0])
        next_tick = first + float(phase_offset_ms)
        index = 0
        current_action = int(MoveAction.IDLE)
        current_escape = False
        escaped_episode = False
        previous_escape_action: int | None = None
        in_window = False
        window_failed = False
        window_actionable_time = None
        window_actionable_tti = None
        last_health = _health(rows[0][0])
        session_dead = False
        last_frame_time = first
        last_decision_action: int | None = None
        last_decision_time: float | None = None
        while index < len(rows):
            snapshot, requested = rows[index]
            now = _timestamp(snapshot)
            payload = _payload(snapshot)
            cached = diagnostics.get(id(snapshot.payload)) if diagnostics is not None else None
            if cached is None:
                actionable, tti, min_risk = _actionable(
                    payload, scorer, baseline_action=requested
                )
                frame_risks = scorer.all_risks(payload)
            else:
                actionable, tti, frame_risks = cached
            if actionable and not in_window:
                result.hazard_windows += 1
                if escaped_episode:
                    result.reentries += 1
                    escaped_episode = False
                in_window = True
                window_failed = False
                window_actionable_time = now
                window_actionable_tti = tti
                # Scheduling delay is measured from the first actionable frame
                # to the next decision tick, including a zero-delay tick.
                next_permitted_tick = _next_tick_at_or_after(now, next_tick, interval)
                result.actionable_delays_ms.append(max(0.0, next_permitted_tick - now))
            if in_window and _health(snapshot) < last_health:
                window_failed = True
                result.health_loss_events += 1
                if not any(
                    event["row_index"] == index for event in result.failure_events
                ):
                    result.failure_events.append(
                        {
                            "row_index": index,
                            "impact_time_ms": now,
                            "actionable_time_ms": window_actionable_time,
                            "actionable_tti_ms": window_actionable_tti,
                        }
                    )
                risks = frame_risks
                current_risk = risks[current_action].total
                best_action = min(risks, key=lambda action: (risks[action].total, action == 0, action))
                best_risk = risks[best_action].total
                if current_risk - best_risk >= 0.08 and best_risk < 0.65:
                    category = "control_tick_arrived_too_late"
                elif best_risk >= 0.65:
                    category = "no_geometrically_safe_action"
                elif tti is None:
                    category = "prediction_or_observation_mismatch"
                else:
                    category = "policy_selected_wrong_action"
                result.failure_categories[category] = result.failure_categories.get(category, 0) + 1
                if tti is not None:
                    result.projectile_hits += 1
                    result.tti_failures_ms.append(tti)
            if not actionable and in_window:
                result.failed_hazard_windows += int(window_failed)
                in_window = False
            result.separations.extend([_enemy_separation(snapshot)] if _enemy_separation(snapshot) is not None else [])
            if snapshot.dead and not session_dead:
                result.death_events += 1
                session_dead = True
            if _health(snapshot) < last_health and not in_window:
                result.health_loss_events += 1
            while next_tick <= now + 1e-6:
                # A source row that arrived at ``now`` is not available to a
                # tick that occurred before ``now``.  Use the most recent
                # observation whose timestamp has arrived; this is the
                # offline equivalent of a latest-state socket consumer.
                chosen_index = index
                if _timestamp(rows[chosen_index][0]) > next_tick + 1e-6:
                    chosen_index = max(0, chosen_index - 1)
                chosen_snapshot, requested_at_tick = rows[chosen_index], rows[chosen_index][1]
                decision_time = next_tick
                trace = pipeline.apply(chosen_snapshot[0], requested_at_tick, previous_action=current_action, control_interval_ms=interval)
                current_action = trace.decision.applied_action
                if last_decision_action is not None:
                    result.action_changes += int(current_action != last_decision_action)
                    old_vector = ACTION_VECTORS[MoveAction(int(last_decision_action))]
                    new_vector = ACTION_VECTORS[MoveAction(int(current_action))]
                    result.action_oscillations += int(
                        current_action != last_decision_action
                        and old_vector[0] * new_vector[0]
                        + old_vector[1] * new_vector[1]
                        <= -0.5
                    )
                if last_decision_time is not None:
                    result.action_hold_ms.append(max(0.0, decision_time - last_decision_time))
                result.stale_observation_ms.append(
                    max(0.0, decision_time - _timestamp(chosen_snapshot[0]))
                )
                last_decision_action = current_action
                last_decision_time = decision_time
                new_escape = trace.source == "crowd_recovery"
                if new_escape and not current_escape:
                    result.escape_entries += 1
                if new_escape and previous_escape_action is not None and current_action != previous_escape_action:
                    result.escape_reversals += 1
                if current_escape and not new_escape:
                    escaped_episode = True
                current_escape = new_escape
                previous_escape_action = current_action if new_escape else previous_escape_action
                decision_risks = (
                    diagnostics[id(chosen_snapshot[0].payload)][2]
                    if diagnostics is not None and id(chosen_snapshot[0].payload) in diagnostics
                    else scorer.all_risks(chosen_snapshot[0].payload)
                )
                safest = min(decision_risks, key=lambda action: (decision_risks[action].total, action == 0, action))
                result.safest_actions += int(current_action == safest)
                result.decisions += 1
                next_tick += interval
            last_health = _health(snapshot)
            dt = max(0.0, now - last_frame_time)
            result.total_time_ms += dt
            result.escape_time_ms += dt * int(current_escape)
            result.ranged_spacing_time_ms += dt * int(_in_desired_spacing(snapshot))
            last_frame_time = now
            index += 1
        if in_window:
            result.failed_hazard_windows += int(window_failed)
        # A re-entry is an actionable hazard that appears after an escape
        # decision ended.  Count it once per escape episode.
        result.reentries += 0
        if rate_hz == 24.0 and diagnostics is not None and result.failure_events:
            result.counterfactual_extra_decision = _counterfactual_extra_decisions(
                rows, rate_hz, diagnostics, result.failure_events
            )
    return result


def _counterfactual_extra_decisions(
    rows: list[tuple[StateSnapshot, int]],
    rate_hz: float,
    diagnostics: Mapping[int, tuple[bool, float | None, dict[int, Any]]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay each failure and insert one extra policy call before impact.

    This is deliberately a replay diagnostic, not a claim that the passive
    recording would have followed a different physical trajectory.  The
    pipeline is rerun from the beginning of the session for each failure so
    the stateful recovery guard has the same history it had at the current
    rate before the inserted decision.
    """

    if not rows:
        return {"tested": 0, "would_have_helped": 0, "categories": {}}
    interval = 1000.0 / float(rate_hz)
    scorer = CombatSafetyShield(enabled=True, switch_penalty=0.05)
    counts: dict[str, int] = {}
    tested = 0
    helped = 0
    for failure in failures:
        impact_index = int(failure["row_index"])
        if impact_index <= 0 or impact_index >= len(rows):
            continue
        impact_snapshot = rows[impact_index][0]
        impact_risks = diagnostics.get(id(impact_snapshot.payload))
        if impact_risks is None:
            continue
        pipeline = _new_pipeline(diagnostics)
        first = _timestamp(rows[0][0])
        next_tick = first
        current_action = int(MoveAction.IDLE)
        latest = rows[0]
        for row_index in range(impact_index):
            latest = rows[row_index]
            now = _timestamp(latest[0])
            while next_tick <= now + 1e-6 and next_tick < _timestamp(impact_snapshot):
                trace = pipeline.apply(
                    latest[0],
                    latest[1],
                    previous_action=current_action,
                    control_interval_ms=interval,
                )
                current_action = trace.decision.applied_action
                next_tick += interval
        # Use the last observation strictly before impact, then insert exactly
        # one extra call at that timestamp.  Candidate quality is evaluated on
        # the actual impact frame so the comparison is against the same threat.
        extra_trace = pipeline.apply(
            latest[0],
            latest[1],
            previous_action=current_action,
            control_interval_ms=interval,
        )
        extra_action = int(extra_trace.decision.applied_action)
        risks = impact_risks[2]
        baseline_risk = float(risks[current_action].total)
        extra_risk = float(risks[extra_action].total)
        best_risk = min(float(risk.total) for risk in risks.values())
        tested += 1
        if extra_risk + 0.08 < baseline_risk and extra_risk < 0.65:
            category = "one_extra_decision_would_have_helped"
            helped += 1
        elif best_risk >= 0.65:
            category = "no_geometrically_safe_action"
        elif extra_action == current_action:
            category = "policy_same_action_with_extra_call"
        else:
            category = "extra_call_not_materially_safer"
        counts[category] = counts.get(category, 0) + 1
    return {"tested": tested, "would_have_helped": helped, "categories": counts}


def _build_diagnostics(records: list[tuple[StateSnapshot, int]]) -> dict[int, tuple[bool, float | None, dict[int, Any]]]:
    diagnostic_scorer = CombatSafetyShield(enabled=True, switch_penalty=0.05)
    diagnostics: dict[int, tuple[bool, float | None, dict[int, Any]]] = {}
    for snapshot, _requested in records:
        payload = _payload(snapshot)
        actionable, tti, _min_risk = _actionable(
            payload, diagnostic_scorer, baseline_action=_requested
        )
        diagnostics[id(snapshot.payload)] = (actionable, tti, diagnostic_scorer.all_risks(payload))
    return diagnostics


def _evaluate_condition_records(task: tuple[list[tuple[StateSnapshot, int]], float, float]) -> dict[str, Any]:
    records, rate_hz, phase_offset_ms = task
    diagnostics = _build_diagnostics(records)
    return run_rate(records, rate_hz, phase_offset_ms, diagnostics=diagnostics).to_dict()


def _evaluate_condition_path(task: tuple[str, float, float]) -> dict[str, Any]:
    path, rate_hz, phase_offset_ms = task
    records = list(JsonlReplay(Path(path)).records())
    if not records:
        raise ValueError(f"recording contains no replayable raw_state rows: {path}")
    return _evaluate_condition_records((records, rate_hz, phase_offset_ms))


def _condition_tasks(rate_hz_values: Iterable[float], phase_offsets: Iterable[float]) -> list[tuple[float, float]]:
    return [(rate, 0.0) for rate in rate_hz_values] + [(15.0, phase) for phase in phase_offsets]


def _evaluate_parallel_path(recording: Path, workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [(str(recording), rate, phase) for rate, phase in _condition_tasks(RATES_HZ, PHASE_OFFSETS_MS)]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        values = list(pool.map(_evaluate_condition_path, tasks))
    return values[:len(RATES_HZ)], values[len(RATES_HZ):]


def _evaluate_parallel_records(records: list[tuple[StateSnapshot, int]], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [(records, rate, phase) for rate, phase in _condition_tasks(RATES_HZ, PHASE_OFFSETS_MS)]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        values = list(pool.map(_evaluate_condition_records, tasks))
    return values[:len(RATES_HZ)], values[len(RATES_HZ):]


def _recording_summary(recording: Path) -> tuple[int, int]:
    sessions: set[str] = set()
    replayable = 0
    with Path(recording).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if raw.get("type") in {"raw_state", "state"} and 0 <= int(raw.get("action", -1)) < 9:
                replayable += 1
                sessions.add(str(raw.get("session", "")))
    return replayable, len(sessions)


def run_experiment(recording: Path, *, workers: int = 1) -> dict[str, Any]:
    workers = max(1, min(len(RATES_HZ) + len(PHASE_OFFSETS_MS), int(workers)))
    if workers == 1:
        records = list(JsonlReplay(recording).records())
        if not records:
            raise ValueError(f"recording contains no replayable raw_state rows: {recording}")
        rate_results, phase_results = _evaluate_records(records)
        input_records, input_sessions = len(records), len(_sessions(records))
    else:
        input_records, input_sessions = _recording_summary(recording)
        if not input_records:
            raise ValueError(f"recording contains no replayable raw_state rows: {recording}")
        rate_results, phase_results = _evaluate_parallel_path(Path(recording), workers)
    return {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": _sha256(recording),
        "input_records": input_records,
        "input_sessions": input_sessions,
        "source_rate_hz": 60,
        "rates": [result if isinstance(result, dict) else result.to_dict() for result in rate_results],
        "phase_offsets_15hz": [result if isinstance(result, dict) else result.to_dict() for result in phase_results],
        "method": "Fixed 60 Hz observations; latest available frame at each scheduled tick; held action between ticks; one unchanged v4 pipeline per session.",
    }


def _evaluate_records(records: list[tuple[StateSnapshot, int]]) -> tuple[list[ReplayResult], list[ReplayResult]]:
    diagnostic_scorer = CombatSafetyShield(enabled=True, switch_penalty=0.05)
    diagnostics: dict[int, tuple[bool, float | None, dict[int, Any]]] = {}
    for snapshot, _requested in records:
        payload = _payload(snapshot)
        actionable, tti, _min_risk = _actionable(
            payload, diagnostic_scorer, baseline_action=_requested
        )
        diagnostics[id(snapshot)] = (actionable, tti, diagnostic_scorer.all_risks(payload))
    rate_results = [run_rate(records, rate, diagnostics=diagnostics) for rate in RATES_HZ]
    phase_results = [run_rate(records, 15.0, phase, diagnostics=diagnostics) for phase in PHASE_OFFSETS_MS]
    return rate_results, phase_results


def estimate_runtime(recording: Path, *, sample_records: int = 500) -> dict[str, Any]:
    """Estimate full-sweep wall time by timing the same nine-condition replay on a sample."""
    recording = Path(recording)
    with recording.open("r", encoding="utf-8") as handle:
        total_records = sum(1 for _ in handle)
    sample_size = min(max(1, int(sample_records)), total_records)
    sample = list(islice(JsonlReplay(recording).records(), sample_size))
    if not sample:
        raise ValueError(f"recording contains no replayable raw_state rows: {recording}")
    started = time.perf_counter()
    workers = max(1, min(len(RATES_HZ) + len(PHASE_OFFSETS_MS), int(os.environ.get("BROTATO_RATE_WORKERS", "1"))))
    # The bounded sample is intentionally timed in-process.  Full workers
    # read the file independently, avoiding Windows spawn/pickle overhead for
    # immutable mapping-proxy snapshots; scale the measured CPU time by the
    # configured batch width for a wall-time estimate.
    _evaluate_records(sample)
    sample_seconds = max(0.001, time.perf_counter() - started)
    estimated_seconds = sample_seconds * total_records / len(sample) / workers
    return {
        "recording": str(recording.resolve()),
        "total_file_records": total_records,
        "replayable_sample_records": len(sample),
        "sample_sweep_seconds": sample_seconds,
        "estimated_full_sweep_seconds": estimated_seconds,
        "estimated_full_sweep_minutes": estimated_seconds / 60.0,
        "conditions": len(RATES_HZ) + len(PHASE_OFFSETS_MS),
        "note": "Estimate times the same sweep on a bounded sample and scales by worker count; actual time varies with enemy/projectile counts, process startup, disk contention, and machine load.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-sample", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(9, os.cpu_count() or 1))
    args = parser.parse_args()
    os.environ["BROTATO_RATE_WORKERS"] = str(max(1, args.workers))
    estimate = estimate_runtime(args.recording, sample_records=max(1, args.estimate_sample))
    print(json.dumps(estimate, indent=2))
    if args.estimate_only:
        return 0
    report = run_experiment(args.recording, workers=max(1, args.workers))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[control-rate] records={report['input_records']} json={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
