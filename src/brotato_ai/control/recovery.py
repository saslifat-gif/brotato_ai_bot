"""Explicit crowd/edge emergency mode that reuses unified hazard scores."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from brotato_ai.control.hazards import UnifiedHazardScorer
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


class CrowdRecoveryGuard:
    """Force a short low-risk escape only under explicit emergency conditions."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        wave_threshold: int = 14,
        enemy_threshold: int = 18,
        boundary_threshold: float = 0.45,
        hold_steps: int = 8,
        shield: UnifiedHazardScorer | None = None,
    ):
        self.enabled = bool(enabled)
        self.wave_threshold = int(wave_threshold)
        self.enemy_threshold = int(enemy_threshold)
        self.boundary_threshold = float(boundary_threshold)
        self.hold_steps = max(1, int(hold_steps))
        self.remaining = 0
        self.shield = shield if shield is not None else UnifiedHazardScorer(enabled=True)

    @staticmethod
    def _payload(state: Mapping[str, Any] | StateSnapshot) -> Mapping[str, Any]:
        return state.to_dict() if isinstance(state, StateSnapshot) else state

    def reset(self) -> None:
        self.remaining = 0

    @staticmethod
    def _center_action(state: Mapping[str, Any] | StateSnapshot) -> int:
        state = CrowdRecoveryGuard._payload(state)
        player = _mapping(state.get("player"))
        arena = _mapping(state.get("arena"))
        px, py = _xy(player.get("position"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        safe_x = min(max(px, width * 0.25), width * 0.75)
        safe_y = min(max(py, height * 0.25), height * 0.75)
        dx = safe_x - px
        dy = safe_y - py
        horizontal = 1 if dx > width * 0.02 else -1 if dx < -width * 0.02 else 0
        vertical = 1 if dy > height * 0.02 else -1 if dy < -height * 0.02 else 0
        if horizontal == 0 and vertical == 0:
            return int(MoveAction.IDLE)
        if horizontal < 0 and vertical < 0:
            return int(MoveAction.UP_LEFT)
        if horizontal > 0 and vertical < 0:
            return int(MoveAction.UP_RIGHT)
        if horizontal < 0 and vertical > 0:
            return int(MoveAction.DOWN_LEFT)
        if horizontal > 0 and vertical > 0:
            return int(MoveAction.DOWN_RIGHT)
        if horizontal < 0:
            return int(MoveAction.LEFT)
        if horizontal > 0:
            return int(MoveAction.RIGHT)
        return int(MoveAction.UP if vertical < 0 else MoveAction.DOWN)

    @staticmethod
    def _safest_escape_action(
        state: Mapping[str, Any] | StateSnapshot,
        shield: UnifiedHazardScorer | None = None,
    ) -> int:
        payload = CrowdRecoveryGuard._payload(state)
        center_action = CrowdRecoveryGuard._center_action(payload)
        shield = shield if shield is not None else UnifiedHazardScorer(
            enabled=True, override_margin=0.0
        )
        player = _mapping(payload.get("player"))
        arena = _mapping(payload.get("arena"))
        px, py = _xy(player.get("position"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        safe_x = min(max(px, width * 0.25), width * 0.75)
        safe_y = min(max(py, height * 0.25), height * 0.75)
        dx = safe_x - px
        dy = safe_y - py
        center_length = max(1.0, math.hypot(dx, dy))
        center_vector = (dx / center_length, dy / center_length)
        scored: list[tuple[float, int]] = []
        for action, movement in ACTION_VECTORS.items():
            if action == MoveAction.IDLE:
                continue
            risk = shield.risk(payload, int(action))
            toward_safe_band = movement[0] * center_vector[0] + movement[1] * center_vector[1]
            center_bias = 0.08 if int(action) == center_action else 0.0
            scored.append((risk - 0.12 * toward_safe_band - center_bias, int(action)))
        return min(scored, key=lambda row: (row[0], row[1]))[1]

    def apply(
        self,
        state: Mapping[str, Any] | StateSnapshot,
        requested_action: int,
        *,
        risks: Mapping[int, HazardRisk] | None = None,
    ) -> SafetyDecision:
        requested = int(MoveAction(int(requested_action)))
        payload = self._payload(state)
        if not self.enabled:
            return SafetyDecision(requested, requested, 0.0, 0.0)
        wave = int(_number(_mapping(payload.get("wave")).get("number"), 0))
        enemy_count = len(_items(payload.get("enemies")))
        boundary = _mapping(payload.get("projectile_paths")).get(
            "boundary_action_risk", []
        )
        boundary_max = (
            max((_number(value) for value in boundary), default=0.0)
            if isinstance(boundary, (list, tuple))
            else 0.0
        )
        trigger = (
            boundary_max >= 0.80
            or (
                wave >= self.wave_threshold
                and (
                    enemy_count >= self.enemy_threshold
                    or boundary_max >= self.boundary_threshold
                )
            )
        )
        if self.remaining <= 0 and trigger:
            self.remaining = self.hold_steps
        if self.remaining <= 0:
            risk = (
                risks[requested].total
                if risks is not None
                else self.shield.risk(payload, requested)
            )
            return SafetyDecision(requested, requested, risk, risk)
        self.remaining -= 1
        if risks is None:
            escape_action = self._safest_escape_action(payload, self.shield)
            requested_risk = self.shield.risk(payload, requested)
            applied_risk = self.shield.risk(payload, escape_action)
        else:
            center_action = self._center_action(payload)
            player = _mapping(payload.get("player"))
            arena = _mapping(payload.get("arena"))
            px, py = _xy(player.get("position"))
            width = max(1.0, _number(arena.get("width"), 1920.0))
            height = max(1.0, _number(arena.get("height"), 1080.0))
            safe_x = min(max(px, width * 0.25), width * 0.75)
            safe_y = min(max(py, height * 0.25), height * 0.75)
            dx, dy = safe_x - px, safe_y - py
            length = max(1.0, math.hypot(dx, dy))
            center_vector = (dx / length, dy / length)
            escape_action = min(
                (int(action) for action in MoveAction if action != MoveAction.IDLE),
                key=lambda action: (
                    risks[action].total
                    - 0.12
                    * (
                        ACTION_VECTORS[MoveAction(action)][0] * center_vector[0]
                        + ACTION_VECTORS[MoveAction(action)][1] * center_vector[1]
                    )
                    - (0.08 if action == center_action else 0.0),
                    action,
                ),
            )
            requested_risk = risks[requested].total
            applied_risk = risks[escape_action].total
        return SafetyDecision(
            requested,
            escape_action,
            requested_risk,
            applied_risk,
        )
