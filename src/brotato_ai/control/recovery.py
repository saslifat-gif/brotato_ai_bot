"""Persistent tactical movement state for the active V4 controller."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from brotato_ai.control.materials import material_progress
from brotato_ai.control.safe_zone import SafeZonePlanner, edge_clearance
from brotato_ai.control.hazards import UnifiedHazardScorer, enemy_separation_diagnostics
from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction
from brotato_ai.domain.decisions import HazardRisk, SafetyDecision
from brotato_ai.domain.state import StateSnapshot


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _xy(value: Any) -> tuple[float, float]:
    item = _mapping(value)
    return _number(item.get("x")), _number(item.get("y"))


class TacticalMovementController:
    """Persistent NORMAL/ESCAPE controller with hysteresis and lateral escape.

    Hold timing is expressed in real-time milliseconds
    (``hold_duration_ms`` / ``side_hold_duration_ms``).  Step counts remain
    supported as a fallback for callers that cannot supply a measured control
    interval, and the defaults reproduce the previous 24 Hz behavior exactly
    (8 steps ~= 333 ms, 6 steps ~= 250 ms at 24 Hz).  At any other control
    rate the duration semantics keep the wall-clock hold stable instead of
    silently scaling with the bridge rate.
    """

    NORMAL = "normal"
    ESCAPE = "escape"

    DEFAULT_CONTROL_HZ = 24.0
    HOLD_TOLERANCE_MS = 1.0

    def __init__(
        self,
        *,
        enabled: bool = True,
        wave_threshold: int = 14,
        enemy_threshold: int = 18,
        boundary_threshold: float = 0.45,
        hold_steps: int = 8,
        hold_duration_ms: float | None = None,
        shield: UnifiedHazardScorer | None = None,
        escape_enter_risk: float = 0.28,
        escape_exit_risk: float = 0.16,
        release_margin: float = 1.15,
        side_hold_steps: int = 6,
        side_hold_duration_ms: float | None = None,
        default_control_hz: float = DEFAULT_CONTROL_HZ,
    ):
        self.safe_zone = SafeZonePlanner()
        self.idle_escape_ms = 0.0
        self.anti_stall_active = False
        self._material_progress = {}
        self._step_geometry = None
        self._step_frame = None
        self.enabled = bool(enabled)
        self.wave_threshold = int(wave_threshold)
        self.enemy_threshold = int(enemy_threshold)
        self.boundary_threshold = float(boundary_threshold)
        self.hold_steps = max(1, int(hold_steps))
        self.hold_duration_ms = (
            float(hold_duration_ms)
            if hold_duration_ms is not None
            else self.hold_steps * 1000.0 / max(1.0, float(default_control_hz))
        )
        self.escape_enter_risk = max(0.0, float(escape_enter_risk))
        self.escape_exit_risk = max(0.0, float(escape_exit_risk))
        self.release_margin = max(1.0, float(release_margin))
        self.side_hold_steps = max(1, int(side_hold_steps))
        self.side_hold_duration_ms = (
            float(side_hold_duration_ms)
            if side_hold_duration_ms is not None
            else self.side_hold_steps * 1000.0 / max(1.0, float(default_control_hz))
        )
        self.shield = shield if shield is not None else UnifiedHazardScorer(enabled=True)
        self.state = self.NORMAL
        self.remaining = 0
        self.escape_side = 0
        self._age = 0
        self._side_age = 0
        self._age_ms = 0.0
        self._side_age_ms = 0.0
        self._saw_control_interval = False
        self._last_escape_action: int | None = None

    @property
    def active(self) -> bool:
        return self.state == self.ESCAPE

    @property
    def state_name(self) -> str:
        return self.state

    def reset(self) -> None:
        self.safe_zone.reset()
        self.idle_escape_ms = 0.0
        self.anti_stall_active = False
        self.state = self.NORMAL
        self.remaining = 0
        self.escape_side = 0
        self._age = 0
        self._side_age = 0
        self._age_ms = 0.0
        self._side_age_ms = 0.0
        self._saw_control_interval = False
        self._last_escape_action = None

    @property
    def remaining_ms(self) -> float:
        """Wall-clock hold time left in the current escape, in milliseconds."""

        if self.state != self.ESCAPE:
            return 0.0
        return max(0.0, self.hold_duration_ms - self._age_ms)

    @staticmethod
    def _payload(state: Mapping[str, Any] | StateSnapshot) -> Mapping[str, Any]:
        return state.payload if isinstance(state, StateSnapshot) else state

    @staticmethod
    def _enemy_frame(payload: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float], float]:
        player = _mapping(payload.get("player"))
        px, py = _xy(player.get("position"))
        enemies = [e for e in _items(payload.get("enemies")) if not bool(e.get("dead"))]
        if not enemies:
            return (0.0, 0.0), (0.0, 0.0), 0.0
        enemy = min(
            enemies,
            key=lambda e: math.hypot(_xy(e.get("position"))[0] - px, _xy(e.get("position"))[1] - py),
        )
        ex, ey = _xy(enemy.get("position"))
        away_x, away_y = px - ex, py - ey
        length = max(1.0, math.hypot(away_x, away_y))
        return (away_x / length, away_y / length), (-away_y / length, away_x / length), length

    def _legacy_trigger(self, payload: Mapping[str, Any]) -> bool:
        wave = int(_number(_mapping(payload.get("wave")).get("number"), 0))
        enemy_count = len(_items(payload.get("enemies")))
        paths = _mapping(payload.get("projectile_paths"))
        boundary = paths.get("boundary_action_risk", [])
        boundary_max = max((_number(v) for v in boundary), default=0.0) if isinstance(boundary, (list, tuple)) else 0.0
        packed_late = wave >= 8 and enemy_count >= self.enemy_threshold
        return (
            boundary_max >= 0.80
            or packed_late
            or (
                wave >= self.wave_threshold
                and boundary_max >= self.boundary_threshold
            )
        )

    def _should_enter(
        self,
        payload: Mapping[str, Any],
        requested_risk: HazardRisk,
        requested_action: int,
        risks: Mapping[int, HazardRisk] | None = None,
    ) -> bool:
        if edge_clearance(payload) < 180.:
            # Turn away before a wall becomes an immediate collision risk.
            px, py = _xy(_mapping(payload.get("player")).get("position"))
            ax, ay = ACTION_VECTORS[requested_action]
            if edge_clearance(payload, (px+ax*80., py+ay*80.)) <= edge_clearance(payload):
                return True
        if self._legacy_trigger(payload) and (risks is None or risks[0].total > self.escape_exit_risk):
            return True
        geometry = self._geometry(payload, requested_action)
        closing = bool(
            geometry["active"]
            and float(geometry["predicted_distance"]) <= float(geometry["target_distance"]) * 1.08
            and float(geometry["closing_rate"]) > -0.08
        )
        if closing:
            return True
        high_requested = (
            requested_risk.total >= self.escape_enter_risk
            or requested_risk.enemy_total >= self.escape_enter_risk
        )
        if not high_requested:
            return False
        if risks is None:
            return True
        best_risk = min(float(risk.total) for risk in risks.values())
        # A safer policy lane exists; let choose() take it instead of ESCAPE.
        return best_risk >= self.escape_enter_risk

    def _geometry(self, payload, action):
        if self._step_geometry is None:
            return enemy_separation_diagnostics(payload, action)
        if action not in self._step_geometry:
            self._step_geometry[action] = enemy_separation_diagnostics(payload, action)
        return self._step_geometry[action]

    def _frame_for_step(self, payload):
        if self._step_geometry is None:
            return self._enemy_frame(payload)
        if self._step_frame is None:
            self._step_frame = self._enemy_frame(payload)
        return self._step_frame

    def _score_action(
        self,
        payload: Mapping[str, Any],
        risks: Mapping[int, HazardRisk],
        action: int,
        *,
        side: int,
        previous_action: int | None,
    ) -> float:
        movement = ACTION_VECTORS[MoveAction(int(action))]
        score = float(risks[int(action)].total)
        geometry = self._geometry(payload, action)
        if geometry["active"]:
            approach = max(0.0, -float(geometry["radial_dot"]))
            closing = max(0.0, float(geometry["closing_rate"]))
            score += (0.85 + 1.35 * closing) * approach
            away, tangent, _ = self._frame_for_step(payload)
            lateral = movement[0] * tangent[0] + movement[1] * tangent[1]
            radial = movement[0] * away[0] + movement[1] * away[1]
            score -= 0.20 * max(0.0, radial)
            score -= 0.16 * max(0.0, float(side) * lateral)
            score -= 0.08 * max(0.0, abs(lateral) - abs(radial))
        if previous_action is not None:
            previous = ACTION_VECTORS[MoveAction(int(previous_action))]
            if movement[0] * previous[0] + movement[1] * previous[1] < -0.70:
                score += 0.16
        if self._last_escape_action is not None and action != self._last_escape_action:
            prior = ACTION_VECTORS[MoveAction(self._last_escape_action)]
            if movement[0] * prior[0] + movement[1] * prior[1] < -0.70:
                score += 0.20
        # Money only breaks close escape choices; separation and risk retain
        # their larger penalties. Compute attraction once per control step.
        if risks[action].total <= min(.20, min(r.total for r in risks.values()) + .03):
            score -= .12 * max(0., self._material_progress.get(action, 0.))
        return score

    def _choose_side(self, payload: Mapping[str, Any], risks: Mapping[int, HazardRisk], previous_action: int | None) -> int:
        enemies = [e for e in _items(payload.get("enemies")) if not bool(e.get("dead"))]
        if not enemies:
            player = _mapping(payload.get("player"))
            arena = _mapping(payload.get("arena"))
            px, py = _xy(player.get("position"))
            width = max(1.0, _number(arena.get("width"), 1920.0))
            height = max(1.0, _number(arena.get("height"), 1080.0))
            return 1 if px / width < py / height else -1
        scores = {
            side: min(
                self._score_action(payload, risks, int(action), side=side, previous_action=previous_action)
                for action in MoveAction if action != MoveAction.IDLE
            )
            for side in (-1, 1)
        }
        return -1 if scores[-1] < scores[1] else 1

    def _escape_action(self, payload: Mapping[str, Any], risks: Mapping[int, HazardRisk], previous_action: int | None) -> int:
        candidates = [int(action) for action in MoveAction if action != MoveAction.IDLE]
        return min(
            candidates,
            key=lambda action: (
                self._score_action(
                    payload,
                    risks,
                    action,
                    side=self.escape_side or 1,
                    previous_action=previous_action,
                ),
                action,
            ),
        )

    def _break_dangerous_idle(self, payload, risks, action, interval_ms):
        self.anti_stall_active = False
        if action != int(MoveAction.IDLE):
            self.idle_escape_ms = 0.0
            return action
        self.idle_escape_ms += max(0.0, interval_ms) or (1000.0 / self.DEFAULT_CONTROL_HZ)
        idle = risks[int(MoveAction.IDLE)]
        if self.idle_escape_ms < 350.0 or idle.enemy_total < self.escape_exit_risk:
            return action
        candidates = []
        for candidate, risk in risks.items():
            if candidate == int(MoveAction.IDLE):
                continue
            # Only a bounded risk increase is allowed, never a blind forced move.
            if (risk.total > idle.total + .35
                    or risk.projectile_total > idle.projectile_total + .02
                    or risk.indicator > idle.indicator + .02
                    or risk.boundary_total > idle.boundary_total + .05
                    or risk.enemy + risk.enemy_path > idle.enemy + idle.enemy_path + .10):
                continue
            geometry = self._geometry(payload, candidate)
            if (geometry["active"]
                    and geometry["predicted_distance"] >= geometry["current_distance"] + 8.0
                    and geometry["radial_dot"] > .05):
                candidates.append(candidate)
        if not candidates:
            return action
        chosen = min(candidates, key=lambda a: (risks[a].total, a))
        self.anti_stall_active = True
        self.idle_escape_ms = 0.0
        return int(chosen)

    def _hold_elapsed(self) -> bool:
        """Minimum escape hold, in real time when intervals are measurable."""

        if self._saw_control_interval:
            return self._age_ms >= self.hold_duration_ms - self.HOLD_TOLERANCE_MS
        return self._age >= self.hold_steps

    def _side_hold_elapsed(self) -> bool:
        if self._saw_control_interval:
            return self._side_age_ms >= self.side_hold_duration_ms - self.HOLD_TOLERANCE_MS
        return self._side_age >= self.side_hold_steps

    def _clear_to_normal(self, payload: Mapping[str, Any], requested_action: int, requested_risk: HazardRisk) -> bool:
        if edge_clearance(payload) < 180.:
            return False
        if not self._hold_elapsed() or requested_risk.total > self.escape_exit_risk:
            return False
        geometry = self._geometry(payload, requested_action)
        if not geometry["active"]:
            return True
        return bool(
            float(geometry["predicted_distance"]) >= float(geometry["target_distance"]) * self.release_margin
            and float(geometry["closing_rate"]) <= 0.02
            and float(geometry["radial_dot"]) >= -0.05
        )

    def apply(self, state, requested_action, *, risks=None, previous_action=None,
              control_interval_ms=0.0):
        # Cache only within this decision. Even reused/mutated input objects
        # and equal ticks must get fresh geometry on the next invocation.
        self._step_geometry = {}
        self._step_frame = None
        try:
            return self._apply_step(
                state, requested_action, risks=risks,
                previous_action=previous_action,
                control_interval_ms=control_interval_ms,
            )
        finally:
            self._step_geometry = None
            self._step_frame = None

    def _apply_step(
        self,
        state: Mapping[str, Any] | StateSnapshot,
        requested_action: int,
        *,
        risks: Mapping[int, HazardRisk] | None = None,
        previous_action: int | None = None,
        control_interval_ms: float = 0.0,
    ) -> SafetyDecision:
        self.anti_stall_active = False
        requested = int(MoveAction(int(requested_action)))
        payload = self._payload(state)
        self._material_progress = material_progress(payload)
        if risks is None:
            risks = self.shield.all_risks(payload)
        requested_risk = risks[requested]
        if not self.enabled:
            return SafetyDecision(requested, requested, requested_risk.total, requested_risk.total)
        if control_interval_ms > 0.0:
            self._saw_control_interval = True
        if self.state == self.NORMAL and self._should_enter(
            payload, requested_risk, requested, risks
        ):
            self.state = self.ESCAPE
            self.remaining = self.hold_steps
            self._age = 0
            self._age_ms = 0.0
            self.escape_side = self._choose_side(payload, risks, previous_action)
            self._side_age = 0
            self._side_age_ms = 0.0
        elif self.state == self.ESCAPE and self._clear_to_normal(payload, requested, requested_risk):
            self.reset()
            return SafetyDecision(requested, requested, requested_risk.total, requested_risk.total)
        if self.state != self.ESCAPE:
            return SafetyDecision(requested, requested, requested_risk.total, requested_risk.total)
        if self._side_hold_elapsed():
            preferred_side = self._choose_side(payload, risks, previous_action)
            if preferred_side != self.escape_side:
                current_action = self._escape_action(payload, risks, previous_action)
                current_score = self._score_action(payload, risks, current_action, side=self.escape_side or 1, previous_action=previous_action)
                new_score = self._score_action(payload, risks, current_action, side=preferred_side, previous_action=previous_action)
                if new_score + 0.25 < current_score:
                    self.escape_side = preferred_side
                    self._side_age = 0
        escape_action = self._escape_action(payload, risks, previous_action)
        safest = min(
            risks,
            key=lambda action: (risks[action].total, action == 0, action),
        )
        if risks[safest].total + 0.12 < risks[escape_action].total:
            escape_action = int(safest)
        escape_action = self._break_dangerous_idle(
            payload, risks, escape_action, control_interval_ms
        )
        escape_action = self.safe_zone.apply(payload, risks, escape_action, control_interval_ms)
        if (self.safe_zone.arrived and self._hold_elapsed()
                and risks[0].total <= self.escape_exit_risk
                and self._clear_to_normal(payload, 0, risks[0])):
            self.reset()
            return SafetyDecision(requested, 0, requested_risk.total, risks[0].total)
        self._last_escape_action = escape_action
        self._age += 1
        self._side_age += 1
        # Accumulate real time with the same placement as the step counter:
        # the entry step consumes one interval, mirroring `_age += 1`.
        self._age_ms += max(0.0, float(control_interval_ms))
        self._side_age_ms += max(0.0, float(control_interval_ms))
        self.remaining = max(0, self.hold_steps - self._age)
        return SafetyDecision(requested, escape_action, requested_risk.total, risks[escape_action].total)


# Compatibility name retained for existing callers and tests.
CrowdRecoveryGuard = TacticalMovementController
