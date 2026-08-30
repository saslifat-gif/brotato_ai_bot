"""Structured combat teacher, safety shield, and rich observation base."""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from v4.protocol import MoveAction


RICH_OBSERVATION_SIZE = 384
RICH_MAX_ENEMIES = 20
RICH_MAX_PROJECTILES = 20
RICH_MAX_PICKUPS = 8
SEMANTIC_OBSERVATION_SIZE = 832
SEMANTIC_MAX_INDICATORS = 10
SEMANTIC_MAX_WEAPONS = 6
FULL_ARENA_GRID_COLUMNS = 10
FULL_ARENA_GRID_ROWS = 6
FULL_ARENA_GRID_CHANNELS = 10
FULL_ARENA_ATTACK_FEATURES = 4
FULL_ARENA_GRID_SIZE = (
    FULL_ARENA_GRID_COLUMNS * FULL_ARENA_GRID_ROWS * FULL_ARENA_GRID_CHANNELS
)
FULL_ARENA_ATTACK_SIZE = RICH_MAX_ENEMIES * FULL_ARENA_ATTACK_FEATURES
FULL_ARENA_OBSERVATION_SIZE = (
    SEMANTIC_OBSERVATION_SIZE + FULL_ARENA_GRID_SIZE + FULL_ARENA_ATTACK_SIZE
)
BULLET_HELL_GRID_COLUMNS = 20
BULLET_HELL_GRID_ROWS = 12
BULLET_HELL_GRID_CHANNELS = 10
BULLET_HELL_GRID_SIZE = (
    BULLET_HELL_GRID_COLUMNS * BULLET_HELL_GRID_ROWS * BULLET_HELL_GRID_CHANNELS
)
BULLET_HELL_PROJECTILE_RISK_SIZE = len(MoveAction)
BULLET_HELL_ENEMY_RISK_SIZE = len(MoveAction)
BULLET_HELL_BOUNDARY_RISK_SIZE = len(MoveAction)
BULLET_HELL_ACTION_RISK_SIZE = (
    BULLET_HELL_PROJECTILE_RISK_SIZE
    + BULLET_HELL_ENEMY_RISK_SIZE
    + BULLET_HELL_BOUNDARY_RISK_SIZE
)
BULLET_HELL_METADATA_SIZE = 2
BULLET_HELL_OBSERVATION_SIZE = (
    FULL_ARENA_OBSERVATION_SIZE
    + BULLET_HELL_GRID_SIZE
    + BULLET_HELL_ACTION_RISK_SIZE
    + BULLET_HELL_METADATA_SIZE
)

ACTION_VECTORS = {
    MoveAction.IDLE: (0.0, 0.0),
    MoveAction.UP: (0.0, -1.0),
    MoveAction.DOWN: (0.0, 1.0),
    MoveAction.LEFT: (-1.0, 0.0),
    MoveAction.RIGHT: (1.0, 0.0),
    MoveAction.UP_LEFT: (-math.sqrt(0.5), -math.sqrt(0.5)),
    MoveAction.UP_RIGHT: (math.sqrt(0.5), -math.sqrt(0.5)),
    MoveAction.DOWN_LEFT: (-math.sqrt(0.5), math.sqrt(0.5)),
    MoveAction.DOWN_RIGHT: (math.sqrt(0.5), math.sqrt(0.5)),
}


def movement_transition_metrics(
    previous_state: Mapping[str, Any],
    state: Mapping[str, Any],
    previous_action: int,
    action: int,
    *,
    state_hz: float,
) -> dict[str, float | bool]:
    """Measure real displacement and rapid action reversal for one API step."""

    active = previous_state.get("phase") == "combat" and state.get("phase") == "combat"
    old_position = _xy(_mapping(previous_state.get("player")).get("position"))
    new_position = _xy(_mapping(state.get("player")).get("position"))
    distance = math.hypot(
        new_position[0] - old_position[0],
        new_position[1] - old_position[1],
    ) if active else 0.0
    speed = max(
        1.0,
        _number(_mapping(previous_state.get("combat")).get("move_speed"), 300.0),
    )
    expected_distance = speed / max(1.0, float(state_hz))
    efficiency = distance / expected_distance if active else 0.0
    old_vector = ACTION_VECTORS[MoveAction(int(previous_action))]
    new_vector = ACTION_VECTORS[MoveAction(int(action))]
    dot = old_vector[0] * new_vector[0] + old_vector[1] * new_vector[1]
    reversal = bool(
        active
        and int(previous_action) != int(MoveAction.IDLE)
        and int(action) != int(MoveAction.IDLE)
        and dot <= -0.5
    )
    low_motion = bool(
        active
        and int(action) != int(MoveAction.IDLE)
        and distance < expected_distance * 0.15
    )
    return {
        "active": active,
        "distance": distance,
        "expected_distance": expected_distance,
        "efficiency": efficiency,
        "reversal": reversal,
        "low_motion": low_motion,
    }


def center_stagnation_signal(
    previous_state: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    threat_risk: float,
    radius: float = 0.12,
    threat_exemption: float = 0.45,
) -> bool:
    """Detect lingering near the exact arena center when danger is low."""

    if (
        not isinstance(previous_state, Mapping)
        or not isinstance(state, Mapping)
        or previous_state.get("phase") != "combat"
        or state.get("phase") != "combat"
        or float(threat_risk) >= float(threat_exemption)
    ):
        return False
    previous_player = _mapping(previous_state.get("player"))
    current_player = _mapping(state.get("player"))
    arena = _mapping(state.get("arena"))
    width = max(1.0, _number(arena.get("width"), 1920.0))
    height = max(1.0, _number(arena.get("height"), 1080.0))
    previous_x, previous_y = _xy(previous_player.get("position"))
    current_x, current_y = _xy(current_player.get("position"))
    previous_radius = math.hypot(previous_x / width - 0.5, previous_y / height - 0.5)
    current_radius = math.hypot(current_x / width - 0.5, current_y / height - 0.5)
    center_radius = max(0.0, float(radius))
    return previous_radius <= center_radius and current_radius <= center_radius


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
    data = _mapping(value)
    return _number(data.get("x")), _number(data.get("y"))


def _relative(item: Mapping[str, Any], px: float, py: float) -> tuple[float, float]:
    x, y = _xy(item.get("position"))
    return x - px, y - py


def _nearest(items: list[Mapping[str, Any]], px: float, py: float):
    return sorted(items, key=lambda item: sum(v * v for v in _relative(item, px, py)))


def _enemy_is_boss(enemy: Mapping[str, Any]) -> bool:
    token = f"{enemy.get('id', '')} {enemy.get('type', '')} {enemy.get('name', '')}".lower()
    return bool(enemy.get("is_boss")) or any(
        term in token for term in ("boss", "summoner", "\u53ec\u5524")
    )


def _enemy_runtime_index(enemies: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(enemy.get("runtime_id")): enemy
        for enemy in enemies
        if str(enemy.get("runtime_id", ""))
    }


def _owner_threat_code(
    attack: Mapping[str, Any],
    enemies_by_runtime: Mapping[str, Mapping[str, Any]],
) -> float:
    """Encode an attack's observed owner without hashing unstable node names."""

    owner = enemies_by_runtime.get(str(attack.get("owner_runtime_id", "")))
    if owner is None:
        return 0.0
    return 1.0 if _enemy_is_boss(owner) else 0.5


_ATTACK_METHOD_EMBEDDINGS = {
    "contact": (-1.0, 0.0),
    "charge": (-0.5, 0.8660254),
    "summon": (0.5, 0.8660254),
    "projectile": (1.0, 0.0),
    "area": (0.0, -1.0),
    "unknown": (0.0, 0.0),
}


def _attack_method_embedding(enemy: Mapping[str, Any]) -> tuple[float, float]:
    method = str(enemy.get("attack_method", "")).strip().lower()
    return _ATTACK_METHOD_EMBEDDINGS.get(method, _ATTACK_METHOD_EMBEDDINGS["unknown"])


def projectile_time_to_impact(
    projectile: Mapping[str, Any],
    player_position: tuple[float, float],
    movement: tuple[float, float] = (0.0, 0.0),
    player_speed: float = 300.0,
) -> tuple[float, float]:
    """Return closest-approach time and distance over the next 0.8 seconds."""

    px, py = player_position
    rx, ry = _relative(projectile, px, py)
    qvx, qvy = _xy(projectile.get("velocity"))
    rvx = qvx - movement[0] * player_speed
    rvy = qvy - movement[1] * player_speed
    speed_sq = rvx * rvx + rvy * rvy
    if speed_sq < 1.0:
        return 0.0, math.hypot(rx, ry)
    closest_time = max(0.0, min(0.8, -(rx * rvx + ry * rvy) / speed_sq))
    closest_x = rx + rvx * closest_time
    closest_y = ry + rvy * closest_time
    return closest_time, math.hypot(closest_x, closest_y)


@dataclass(frozen=True)
class SafetyDecision:
    requested_action: int
    applied_action: int
    requested_risk: float
    applied_risk: float

    @property
    def overridden(self) -> bool:
        return self.requested_action != self.applied_action


@dataclass(frozen=True)
class HazardRisk:
    """Inspectable risk breakdown used by the single runtime hazard stage."""

    enemy: float = 0.0
    projectile: float = 0.0
    indicator: float = 0.0
    boundary: float = 0.0
    enemy_path: float = 0.0
    projectile_path: float = 0.0
    boundary_path: float = 0.0

    @property
    def total(self) -> float:
        return float(
            self.enemy
            + self.projectile
            + self.indicator
            + self.boundary
            + self.enemy_path
            + self.projectile_path
            + self.boundary_path
        )

    @property
    def path(self) -> float:
        return float(self.enemy_path + self.projectile_path + self.boundary_path)

    @property
    def enemy_total(self) -> float:
        return float(self.enemy + self.enemy_path)

    @property
    def projectile_total(self) -> float:
        return float(self.projectile + self.projectile_path)

    @property
    def boundary_total(self) -> float:
        return float(self.boundary + self.boundary_path)


@dataclass(frozen=True)
class ProjectileHazardDecision:
    """Decision and diagnostics for the short-horizon projectile selector."""

    requested_action: int
    applied_action: int
    best_action: int
    requested_score: float
    applied_score: float
    best_score: float
    requested_collision_risk: float
    applied_collision_risk: float
    requested_hazard_count: int
    applied_hazard_count: int
    nearest_hazard_tti: float
    overridden: bool
    held: bool = False


class ProjectileHazardSelector:
    """Override imminent projectile hazards when a materially safer lane exists.

    This is intentionally narrower than CombatSafetyShield.  It uses the raw
    hostile projectile geometry and only intervenes when a projectile is on a
    near-term collision course and the score improvement exceeds a margin.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        tti_limit: float = 0.40,
        override_margin: float = 0.05,
        hold_steps: int = 1,
        switch_penalty: float = 0.05,
    ):
        self.enabled = bool(enabled)
        self.tti_limit = max(0.0, float(tti_limit))
        self.override_margin = max(0.0, float(override_margin))
        self.hold_steps = max(1, int(hold_steps))
        self.switch_penalty = max(0.0, float(switch_penalty))
        self._held_action: int | None = None
        self._hold_remaining = 0

    def reset(self) -> None:
        self._held_action = None
        self._hold_remaining = 0

    @staticmethod
    def _hostile_projectiles(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [
            projectile
            for projectile in _items(state.get("projectiles"))
            if "hostile" not in projectile or bool(projectile.get("hostile"))
        ]

    @staticmethod
    def _collision_metrics(
        state: Mapping[str, Any], action: int
    ) -> dict[str, float | int | None]:
        player = _mapping(state.get("player"))
        player_position = _xy(player.get("position"))
        player_speed = max(
            150.0,
            _number(_mapping(state.get("combat")).get("move_speed"), 300.0),
        )
        collision = 0.0
        clearance = 0.0
        tti_term = 0.0
        hazards = 0
        nearest_hazard_tti = None
        projectiles = ProjectileHazardSelector._hostile_projectiles(state)
        movement = ACTION_VECTORS[MoveAction(int(action))]
        for projectile in projectiles:
            tti, miss_distance = projectile_time_to_impact(
                projectile,
                player_position,
                movement,
                player_speed,
            )
            limit = max(8.0, _number(projectile.get("radius"), 12.0)) + 42.0
            urgency = max(0.0, 1.0 - tti / 0.8)
            proximity = max(0.0, 1.0 - miss_distance / limit)
            near_proximity = max(0.0, 1.0 - miss_distance / (limit * 2.0))
            collision += urgency * proximity
            clearance += urgency * near_proximity
            if miss_distance <= limit * 2.0:
                tti_term += urgency
            if tti <= 0.8 and miss_distance <= limit:
                hazards += 1
                if nearest_hazard_tti is None or tti < nearest_hazard_tti:
                    nearest_hazard_tti = float(tti)
        return {
            "collision": collision,
            "clearance": clearance,
            "tti": tti_term,
            "hazards": hazards,
            "nearest_hazard_tti": nearest_hazard_tti,
        }

    def _score(
        self,
        metrics: Mapping[str, float | int | None],
        action: int,
        previous_action: int,
    ) -> float:
        return (
            float(metrics["collision"])
            + 0.35 * float(metrics["clearance"])
            + 0.15 * float(metrics["tti"])
            + (self.switch_penalty if int(action) != int(previous_action) else 0.0)
        )

    def apply(
        self,
        state: Mapping[str, Any],
        requested_action: int,
        *,
        previous_action: int,
    ) -> ProjectileHazardDecision:
        requested = int(MoveAction(int(requested_action)))
        previous = int(MoveAction(int(previous_action)))
        metrics = {
            int(action): self._collision_metrics(state, int(action))
            for action in MoveAction
        }
        scores = {
            action: self._score(metrics[action], action, previous)
            for action in metrics
        }
        best_action = min(
            scores,
            key=lambda action: (scores[action], action != previous, action == int(MoveAction.IDLE)),
        )
        all_ttis = [
            float(item["nearest_hazard_tti"])
            for item in metrics.values()
            if item["nearest_hazard_tti"] is not None
        ]
        nearest_tti = min(all_ttis) if all_ttis else -1.0
        held = False
        if self._hold_remaining > 0 and self._held_action is not None:
            chosen = self._held_action
            self._hold_remaining -= 1
            held = True
        else:
            chosen = requested
            should_override = (
                self.enabled
                and nearest_tti >= 0.0
                and nearest_tti <= self.tti_limit
                and best_action != requested
                and scores[requested] - scores[best_action] > self.override_margin
            )
            if should_override:
                chosen = best_action
                self._held_action = best_action
                self._hold_remaining = max(0, self.hold_steps - 1)
            else:
                self._held_action = None
                self._hold_remaining = 0
        return ProjectileHazardDecision(
            requested_action=requested,
            applied_action=int(chosen),
            best_action=int(best_action),
            requested_score=float(scores[requested]),
            applied_score=float(scores[int(chosen)]),
            best_score=float(scores[best_action]),
            requested_collision_risk=float(metrics[requested]["collision"]),
            applied_collision_risk=float(metrics[int(chosen)]["collision"]),
            requested_hazard_count=int(metrics[requested]["hazards"]),
            applied_hazard_count=int(metrics[int(chosen)]["hazards"]),
            nearest_hazard_tti=float(nearest_tti),
            overridden=int(chosen) != requested,
            held=held,
        )


class CombatSafetyShield:
    """Override actions that are unsafe geometrically or on API path maps."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        override_margin: float = 0.08,
        hard_risk_threshold: float = 0.65,
        minimum_risk: float = 0.22,
        switch_penalty: float = 0.05,
    ):
        self.enabled = bool(enabled)
        self.override_margin = float(override_margin)
        self.hard_risk_threshold = float(hard_risk_threshold)
        self.minimum_risk = max(0.0, float(minimum_risk))
        self.switch_penalty = max(0.0, float(switch_penalty))

    def risk_breakdown(self, state: Mapping[str, Any], action: int) -> HazardRisk:
        player = _mapping(state.get("player"))
        px, py = _xy(player.get("position"))
        arena = _mapping(state.get("arena"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        movement = ACTION_VECTORS[MoveAction(int(action))]
        combat = _mapping(state.get("combat"))
        player_speed = max(150.0, _number(combat.get("move_speed"), 300.0))
        future_x = px + movement[0] * player_speed * 0.45
        future_y = py + movement[1] * player_speed * 0.45
        enemy_risk = 0.0
        projectile_risk = 0.0
        indicator_risk = 0.0
        boundary_risk = 0.0
        projectile_path_risk = 0.0
        enemy_path_risk = 0.0
        boundary_path_risk = 0.0

        edge_margin = 150.0
        edge_distance = min(future_x, width - future_x, future_y, height - future_y)
        if edge_distance < edge_margin:
            boundary_risk += ((edge_margin - edge_distance) / edge_margin) ** 2 * 4.0

        enemies = _items(state.get("enemies"))
        enemies_by_runtime = _enemy_runtime_index(enemies)
        for enemy in _nearest(enemies, px, py)[:24]:
            ex, ey = _xy(enemy.get("position"))
            evx, evy = _xy(enemy.get("velocity"))
            predicted_x = ex + evx * 0.45
            predicted_y = ey + evy * 0.45
            distance = math.hypot(predicted_x - future_x, predicted_y - future_y)
            radius = max(25.0, _number(enemy.get("radius"), 40.0))
            attack_method = str(enemy.get("attack_method", "unknown")).lower()
            charging = bool(enemy.get("is_charging")) or attack_method == "charge"
            method_buffer = {
                "charge": 240.0,
                "area": 190.0,
                "contact": 120.0,
                "summon": 110.0,
                "projectile": 90.0,
            }.get(attack_method, 100.0)
            danger = radius + method_buffer
            if _enemy_is_boss(enemy):
                danger += 60.0
            if distance < danger:
                enemy_risk += ((danger - distance) / danger) ** 2 * (
                    6.0 if charging else 2.0
                )

        wave_data = _mapping(state.get("wave"))
        wave_number = int(_number(wave_data.get("number"), 0.0))
        boss_present = any(_enemy_is_boss(enemy) for enemy in enemies)
        # Brotato's final wave is the boss encounter.  Some bridge versions
        # omit the boss flag, so active hazards on wave 20 must still enable
        # the boss-spacing and projectile-avoidance behavior.
        if not boss_present and wave_number >= 20:
            boss_present = bool(
                _items(state.get("attack_indicators"))
                or _items(state.get("projectiles"))
            )
        if boss_present:
            # During boss attacks, remaining in melee range is more dangerous
            # than the normal crowd estimate suggests.
            player_radius = max(18.0, _number(player.get("radius"), 28.0))
            for enemy in enemies:
                if not _enemy_is_boss(enemy):
                    continue
                ex, ey = _xy(enemy.get("position"))
                boss_distance = math.hypot(ex - future_x, ey - future_y)
                boss_radius = max(45.0, _number(enemy.get("radius"), 55.0))
                # Keep a generous buffer around the actual bodies; the visual
                # sprite and attack body are larger than the center point.
                separation = max(480.0, boss_radius + player_radius + 300.0)
                if boss_distance < separation:
                    enemy_risk += ((separation - boss_distance) / separation) ** 2 * 10.0

        for projectile in _nearest(_items(state.get("projectiles")), px, py)[:32]:
            tti, miss_distance = projectile_time_to_impact(
                projectile, (px, py), movement, player_speed
            )
            radius = max(8.0, _number(projectile.get("radius"), 12.0)) + 42.0
            qvx, qvy = _xy(projectile.get("velocity"))
            stationary = qvx * qvx + qvy * qvy < 100.0
            danger = radius * (2.3 if stationary else 1.5)
            owner_is_boss = _owner_threat_code(projectile, enemies_by_runtime) >= 1.0
            if miss_distance < danger:
                urgency = 1.0 if stationary else 1.0 - min(1.0, tti / 0.8)
                projectile_risk += ((danger - miss_distance) / danger) ** 2 * (
                    (7.0 + 10.0 * urgency)
                    if owner_is_boss
                    else (6.0 + 8.0 * urgency)
                    if boss_present
                    else (3.0 + 5.0 * urgency)
                )

        # Boss telegraphs are stationary warning zones rather than moving
        # projectiles.  Treat an imminent AOE indicator as a hard hazard so a
        # melee policy cannot keep farming loot inside overlapping red circles.
        for indicator in _nearest(_items(state.get("attack_indicators")), px, py)[:32]:
            token = f"{indicator.get('id', '')} {indicator.get('type', '')}".lower()
            owner_is_boss = _owner_threat_code(indicator, enemies_by_runtime) >= 1.0
            if not owner_is_boss and not boss_present and not any(term in token for term in ("aoe", "warning", "circle", "boss")):
                continue
            ix, iy = _xy(indicator.get("position"))
            half_width = max(35.0, _number(indicator.get("width"), 80.0) * 0.5) + 45.0
            half_height = max(35.0, _number(indicator.get("height"), 80.0) * 0.5) + 45.0
            dx = abs(future_x - ix)
            dy = abs(future_y - iy)
            inside = dx <= half_width and dy <= half_height
            time_to_activate = max(0.0, _number(indicator.get("time_to_activate"), 5.0))
            imminent = bool(indicator.get("active")) or time_to_activate <= 1.25
            if inside:
                indicator_risk += (18.0 if imminent else 9.0) if owner_is_boss else (14.0 if imminent else 7.0)
            else:
                gap_x = max(0.0, dx - half_width)
                gap_y = max(0.0, dy - half_height)
                distance = math.hypot(gap_x, gap_y)
                if distance < 180.0:
                    indicator_risk += (1.0 - distance / 180.0) ** 2 * (5.0 if imminent else 2.0)

        # The bridge already predicts enemy, projectile, and boundary paths for
        # every action.  The old shield ignored those vectors, so it could
        # approve an action that the API had explicitly marked dangerous.
        paths = _mapping(state.get("projectile_paths"))
        path_values = (
            ("action_risk", 1.5, "projectile"),
            ("enemy_action_risk", 1.5, "enemy"),
            ("boundary_action_risk", 1.25, "boundary"),
        )
        for key, scale, component in path_values:
            values = paths.get(key, [])
            if not isinstance(values, (list, tuple)) or int(action) >= len(values):
                continue
            value = max(0.0, min(1.0, _number(values[int(action)]))) * scale
            if component == "projectile":
                projectile_path_risk += value
            elif component == "enemy":
                enemy_path_risk += value
            else:
                boundary_path_risk += value
        return HazardRisk(
            enemy=float(enemy_risk),
            projectile=float(projectile_risk),
            indicator=float(indicator_risk),
            boundary=float(boundary_risk),
            enemy_path=float(enemy_path_risk),
            projectile_path=float(projectile_path_risk),
            boundary_path=float(boundary_path_risk),
        )

    def risk(self, state: Mapping[str, Any], action: int) -> float:
        """Return the total risk used by every runtime decision stage."""

        return self.risk_breakdown(state, action).total

    def apply(
        self,
        state: Mapping[str, Any],
        requested_action: int,
        *,
        previous_action: int | None = None,
    ) -> SafetyDecision:
        requested = int(MoveAction(int(requested_action)))
        if not self.enabled:
            return SafetyDecision(requested, requested, 0.0, 0.0)
        requested_risk = self.risk(state, requested)
        if requested_risk < self.minimum_risk:
            return SafetyDecision(requested, requested, requested_risk, requested_risk)
        risks = {int(action): self.risk(state, int(action)) for action in MoveAction}
        previous = (
            int(MoveAction(int(previous_action)))
            if previous_action is not None
            else None
        )
        best_action = min(
            risks,
            key=lambda action: (
                risks[action]
                + (self.switch_penalty if previous is not None and action != previous else 0.0),
                action == int(MoveAction.IDLE),
                action,
            ),
        )
        best_risk = risks[best_action]
        if (
            requested_risk < self.hard_risk_threshold
            and requested_risk - best_risk < self.override_margin
        ):
            best_action, best_risk = requested, requested_risk
        return SafetyDecision(requested, best_action, requested_risk, best_risk)


class EnemyContactGuard:
    """Veto only actions advertised as an imminent enemy-contact path."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        risk_threshold: float = 0.22,
        improvement_margin: float = 0.08,
    ):
        self.enabled = bool(enabled)
        self.risk_threshold = float(risk_threshold)
        self.improvement_margin = float(improvement_margin)

    @staticmethod
    def risks(state: Mapping[str, Any]) -> list[float]:
        paths = _mapping(state.get("projectile_paths"))
        values = paths.get("enemy_action_risk", [])
        output = [0.0] * len(MoveAction)
        if not isinstance(values, (list, tuple)):
            return output
        for index, value in enumerate(values[:len(output)]):
            output[index] = max(0.0, min(1.0, _number(value)))
        return output

    def apply(self, state: Mapping[str, Any], requested_action: int) -> SafetyDecision:
        requested = int(MoveAction(int(requested_action)))
        risks = self.risks(state)
        requested_risk = risks[requested]
        if not self.enabled or requested_risk < self.risk_threshold:
            return SafetyDecision(requested, requested, requested_risk, requested_risk)
        best_action = min(
            range(len(risks)),
            key=lambda action: (risks[action], action == int(MoveAction.IDLE)),
        )
        best_risk = risks[best_action]
        if requested_risk - best_risk < self.improvement_margin:
            return SafetyDecision(requested, requested, requested_risk, requested_risk)
        return SafetyDecision(requested, best_action, requested_risk, best_risk)


class CrowdRecoveryGuard:
    """Force a short low-risk escape before a crowd becomes lethal."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        wave_threshold: int = 14,
        enemy_threshold: int = 18,
        boundary_threshold: float = 0.45,
        hold_steps: int = 8,
        shield: CombatSafetyShield | None = None,
    ):
        self.enabled = bool(enabled)
        self.wave_threshold = int(wave_threshold)
        self.enemy_threshold = int(enemy_threshold)
        self.boundary_threshold = float(boundary_threshold)
        self.hold_steps = max(1, int(hold_steps))
        self.remaining = 0
        self.shield = shield if shield is not None else CombatSafetyShield(enabled=True)

    def reset(self) -> None:
        self.remaining = 0

    @staticmethod
    def _center_action(state: Mapping[str, Any]) -> int:
        """Return an inward action only when outside the safe arena band."""

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
        state: Mapping[str, Any], shield: CombatSafetyShield | None = None
    ) -> int:
        """Choose an inward, moving action without walking through a hazard."""

        center_action = CrowdRecoveryGuard._center_action(state)
        shield = shield if shield is not None else CombatSafetyShield(enabled=True, override_margin=0.0)
        player = _mapping(state.get("player"))
        arena = _mapping(state.get("arena"))
        px, py = _xy(player.get("position"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        safe_x = min(max(px, width * 0.25), width * 0.75)
        safe_y = min(max(py, height * 0.25), height * 0.75)
        dx = safe_x - px
        dy = safe_y - py
        center_length = max(1.0, math.hypot(dx, dy))
        center_vector = (dx / center_length, dy / center_length)
        scored = []
        for action, movement in ACTION_VECTORS.items():
            if action == int(MoveAction.IDLE):
                continue
            risk = shield.risk(state, action)
            toward_safe_band = movement[0] * center_vector[0] + movement[1] * center_vector[1]
            # Prefer an inward direction only when outside the safe band.
            center_bias = 0.08 if action == center_action else 0.0
            scored.append((risk - 0.12 * toward_safe_band - center_bias, action))
        return min(scored, key=lambda row: (row[0], row[1]))[1]

    def apply(self, state: Mapping[str, Any], requested_action: int) -> SafetyDecision:
        requested = int(MoveAction(int(requested_action)))
        if not self.enabled:
            return SafetyDecision(requested, requested, 0.0, 0.0)
        wave = int(_number(_mapping(state.get("wave")).get("number"), 0))
        enemy_count = len(_items(state.get("enemies")))
        paths = _mapping(state.get("projectile_paths"))
        boundary = paths.get("boundary_action_risk", [])
        boundary_max = (
            max((_number(value) for value in boundary), default=0.0)
            if isinstance(boundary, (list, tuple))
            else 0.0
        )
        # The previous three-way AND missed the actual failure mode: waves
        # 14-17 often have either a dense crowd or an edge trap, but not both.
        # Also protect against an extreme edge risk at any wave.
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
            requested_risk = self.shield.risk(state, requested)
            return SafetyDecision(requested, requested, requested_risk, requested_risk)
        self.remaining -= 1
        escape_action = self._safest_escape_action(state, self.shield)
        requested_risk = self.shield.risk(state, requested)
        applied_risk = self.shield.risk(state, escape_action)
        return SafetyDecision(
            requested,
            escape_action,
            requested_risk,
            applied_risk,
        )


@dataclass(frozen=True)
class CombatDecisionTrace:
    """One auditable record of the active action-resolution pipeline."""

    decision: SafetyDecision
    hazard_decision: SafetyDecision
    recovery_decision: SafetyDecision
    requested_risk: HazardRisk
    hazard_risk: HazardRisk
    applied_risk: HazardRisk
    source: str
    enemy_contact_overridden: bool

    @property
    def hazard_overridden(self) -> bool:
        return self.hazard_decision.overridden

    @property
    def recovery_overridden(self) -> bool:
        return self.recovery_decision.overridden


class CombatDecisionPipeline:
    """Resolve one policy action through one unified hazard score.

    The policy proposes an action.  The shared safety shield scores enemy
    movement/contact, projectiles, telegraphs, and boundaries together.  The
    crowd guard is an explicit emergency mode that reuses that same shield;
    it is not a second independent hazard scorer.  This makes precedence and
    telemetry deterministic: hazard first, recovery second, then send once.
    """

    def __init__(
        self,
        *,
        safety_shield: CombatSafetyShield,
        crowd_recovery_guard: CrowdRecoveryGuard,
    ):
        self.safety_shield = safety_shield
        self.crowd_recovery_guard = crowd_recovery_guard

    def reset(self) -> None:
        self.crowd_recovery_guard.reset()

    def apply(
        self,
        state: Mapping[str, Any],
        requested_action: int,
        *,
        previous_action: int | None = None,
    ) -> CombatDecisionTrace:
        requested = int(MoveAction(int(requested_action)))
        requested_risk = self.safety_shield.risk_breakdown(state, requested)
        hazard_decision = self.safety_shield.apply(
            state, requested, previous_action=previous_action
        )
        hazard_risk = self.safety_shield.risk_breakdown(
            state, hazard_decision.applied_action
        )
        recovery_decision = self.crowd_recovery_guard.apply(
            state, hazard_decision.applied_action
        )
        applied_risk = self.safety_shield.risk_breakdown(
            state, recovery_decision.applied_action
        )
        decision = SafetyDecision(
            requested,
            recovery_decision.applied_action,
            requested_risk.total,
            applied_risk.total,
        )
        enemy_contact_overridden = bool(
            decision.overridden
            and requested_risk.enemy_total - applied_risk.enemy_total >= 0.08
        )
        source = (
            "crowd_recovery"
            if recovery_decision.overridden
            else "hazard"
            if hazard_decision.overridden
            else "policy"
        )
        return CombatDecisionTrace(
            decision=decision,
            hazard_decision=hazard_decision,
            recovery_decision=recovery_decision,
            requested_risk=requested_risk,
            hazard_risk=hazard_risk,
            applied_risk=applied_risk,
            source=source,
            enemy_contact_overridden=enemy_contact_overridden,
        )


class CombatHeuristicTeacher:
    """Potential-field teacher used for data collection, not final game strategy."""

    def __init__(self):
        self.shield = CombatSafetyShield(enabled=True, override_margin=0.0)

    def select(self, state: Mapping[str, Any]) -> int:
        player = _mapping(state.get("player"))
        px, py = _xy(player.get("position"))
        health = _number(player.get("health"))
        max_health = max(1.0, _number(player.get("max_health"), 1.0))
        hp_fraction = health / max_health
        desired_x = desired_y = 0.0
        combat = _mapping(state.get("combat"))
        preferred = max(110.0, _number(combat.get("weapon_range"), 170.0))

        for enemy in _nearest(_items(state.get("enemies")), px, py)[:20]:
            dx, dy = _relative(enemy, px, py)
            distance = max(1.0, math.hypot(dx, dy))
            direction_x, direction_y = dx / distance, dy / distance
            if distance < preferred:
                strength = (preferred - distance) / preferred * (3.0 if hp_fraction < 0.5 else 1.5)
                desired_x -= direction_x * strength
                desired_y -= direction_y * strength
            elif distance < preferred * 2.5:
                desired_x += direction_x * 0.08
                desired_y += direction_y * 0.08
            else:
                desired_x += direction_x * 0.03
                desired_y += direction_y * 0.03

        for pickup in _nearest(_items(state.get("pickups")), px, py)[:8]:
            dx, dy = _relative(pickup, px, py)
            distance = max(1.0, math.hypot(dx, dy))
            weight = 0.7 if str(pickup.get("kind")) == "consumable" and hp_fraction < 0.7 else 0.15
            desired_x += dx / distance * weight
            desired_y += dy / distance * weight

        scored = []
        length = max(1e-6, math.hypot(desired_x, desired_y))
        for action, movement in ACTION_VECTORS.items():
            alignment = (movement[0] * desired_x + movement[1] * desired_y) / length
            score = alignment - self.shield.risk(state, int(action)) * 4.0
            scored.append((score, int(action)))
        return max(scored, key=lambda row: (row[0], row[1] != 0))[1]


class RichCombatVectorizer:
    """Versioned 384-value observation for the next combat policy generation."""

    observation_size = RICH_OBSERVATION_SIZE

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray:
        output = np.zeros(RICH_OBSERVATION_SIZE, dtype=np.float32)
        arena = _mapping(state.get("arena"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        player = _mapping(state.get("player"))
        px, py = _xy(player.get("position"))
        pvx, pvy = _xy(player.get("velocity"))
        health = _number(player.get("health"))
        max_health = max(1.0, _number(player.get("max_health"), 1.0))
        wave = _mapping(state.get("wave"))
        counters = _mapping(state.get("counters"))
        combat = _mapping(state.get("combat"))
        enemies = _items(state.get("enemies"))
        projectiles = _items(state.get("projectiles"))
        pickups = _items(state.get("pickups"))
        enemies_by_runtime = _enemy_runtime_index(enemies)
        output[:16] = np.asarray([
            np.clip(px / width * 2.0 - 1.0, -1.0, 1.0),
            np.clip(py / height * 2.0 - 1.0, -1.0, 1.0),
            np.clip(pvx / 1000.0, -1.0, 1.0),
            np.clip(pvy / 1000.0, -1.0, 1.0),
            np.clip(health / max_health, 0.0, 1.0),
            np.clip(_number(wave.get("time_left")) / max(1.0, _number(wave.get("duration"), 60.0)), 0.0, 1.0),
            np.clip(_number(wave.get("number")) / 20.0, 0.0, 1.0),
            np.clip(_number(counters.get("materials")) / 500.0, 0.0, 1.0),
            np.clip(len(enemies) / 128.0, 0.0, 1.0),
            np.clip(len(projectiles) / 128.0, 0.0, 1.0),
            np.clip(len(pickups) / 64.0, 0.0, 1.0),
            1.0 if state.get("phase") == "combat" else 0.0,
            1.0 if state.get("dead") else 0.0,
            1.0 if state.get("victory") else 0.0,
            0.0,
            1.0,
        ], dtype=np.float32)
        if 0 <= int(previous_action) < 9:
            output[16 + int(previous_action)] = 1.0
        weapon_count = max(0.0, _number(combat.get("weapon_count")))
        output[25:32] = np.asarray([
            np.clip(_number(combat.get("weapon_range")) / 1000.0, 0.0, 1.0),
            np.clip(weapon_count / 6.0, 0.0, 1.0),
            np.clip(_number(combat.get("melee_count")) / max(1.0, weapon_count), 0.0, 1.0),
            np.clip(_number(combat.get("ranged_count")) / max(1.0, weapon_count), 0.0, 1.0),
            np.clip(_number(combat.get("move_speed")) / 1000.0, 0.0, 1.0),
            np.clip(_number(combat.get("armor")) / 100.0, -1.0, 1.0),
            np.clip(_number(combat.get("attack_speed")) / 200.0, -1.0, 1.0),
        ], dtype=np.float32)

        cursor = 32
        for enemy in _nearest(enemies, px, py)[:RICH_MAX_ENEMIES]:
            dx, dy = _relative(enemy, px, py)
            vx, vy = _xy(enemy.get("velocity"))
            enemy_max = max(1.0, _number(enemy.get("max_health"), 1.0))
            output[cursor:cursor + 8] = (
                np.clip(dx / width, -1.0, 1.0), np.clip(dy / height, -1.0, 1.0),
                np.clip(vx / 1000.0, -1.0, 1.0), np.clip(vy / 1000.0, -1.0, 1.0),
                np.clip(_number(enemy.get("health")) / enemy_max, 0.0, 1.0),
                np.clip(_number(enemy.get("radius"), 40.0) / 300.0, 0.0, 1.0),
                1.0 if enemy.get("is_boss") else 0.0,
                1.0 if enemy.get("is_charging") else 0.0,
            )
            cursor += 8

        cursor = 32 + RICH_MAX_ENEMIES * 8
        for projectile in _nearest(projectiles, px, py)[:RICH_MAX_PROJECTILES]:
            dx, dy = _relative(projectile, px, py)
            vx, vy = _xy(projectile.get("velocity"))
            tti, miss = projectile_time_to_impact(projectile, (px, py))
            radius = _number(projectile.get("radius"), 12.0)
            owner_code = _owner_threat_code(projectile, enemies_by_runtime)
            output[cursor:cursor + 8] = (
                np.clip(dx / width, -1.0, 1.0), np.clip(dy / height, -1.0, 1.0),
                np.clip(vx / 1000.0, -1.0, 1.0), np.clip(vy / 1000.0, -1.0, 1.0),
                np.clip(radius / 300.0, 0.0, 1.0), np.clip(tti / 0.8, 0.0, 1.0),
                np.clip(miss / 500.0, 0.0, 1.0),
                owner_code if owner_code > 0.0 else (-1.0 if vx * vx + vy * vy < 100.0 else 0.0),
            )
            cursor += 8

        cursor = 32 + RICH_MAX_ENEMIES * 8 + RICH_MAX_PROJECTILES * 8
        for pickup in _nearest(pickups, px, py)[:RICH_MAX_PICKUPS]:
            dx, dy = _relative(pickup, px, py)
            output[cursor:cursor + 4] = (
                np.clip(dx / width, -1.0, 1.0), np.clip(dy / height, -1.0, 1.0),
                1.0 if pickup.get("kind") == "consumable" else 0.5,
                np.clip(math.hypot(dx / width, dy / height), 0.0, 1.0),
            )
            cursor += 4
        return output


def _hash_pair(*values: Any) -> tuple[float, float]:
    token = "|".join(str(value).lower() for value in values if value)
    digest = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
    return (
        (float(digest & 0xFFFF) / 32767.5) - 1.0,
        (float((digest >> 16) & 0xFFFF) / 32767.5) - 1.0,
    )


class SemanticCombatVectorizer:
    """Semantic observation preserving the established feature contract."""

    observation_size = SEMANTIC_OBSERVATION_SIZE

    def __init__(self):
        self.base = RichCombatVectorizer()

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray:
        output = np.zeros(SEMANTIC_OBSERVATION_SIZE, dtype=np.float32)
        output[:RICH_OBSERVATION_SIZE] = self.base.build(state, previous_action)
        arena = _mapping(state.get("arena"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        player = _mapping(state.get("player"))
        px, py = _xy(player.get("position"))
        max_health = max(1.0, _number(player.get("max_health"), 1.0))
        cursor = RICH_OBSERVATION_SIZE

        for enemy in _nearest(_items(state.get("enemies")), px, py)[:RICH_MAX_ENEMIES]:
            identity = _hash_pair(enemy.get("id"), enemy.get("type"))
            attack_method = _attack_method_embedding(enemy)
            output[cursor:cursor + 10] = (
                identity[0], identity[1],
                np.clip(_number(enemy.get("width"), 80.0) / width, 0.0, 1.0),
                np.clip(_number(enemy.get("height"), 80.0) / height, 0.0, 1.0),
                np.clip(_number(enemy.get("contact_damage")) / max_health, 0.0, 4.0) / 4.0,
                np.clip(_number(enemy.get("attack_cooldown_remaining")) / 5.0, 0.0, 1.0),
                1.0 if enemy.get("is_attacking") else 0.0,
                1.0 if enemy.get("is_elite") else 0.0,
                attack_method[0],
                attack_method[1],
            )
            cursor += 10

        cursor = RICH_OBSERVATION_SIZE + RICH_MAX_ENEMIES * 10
        categories = ("healing", "crate", "material", "consumable")
        for pickup in _nearest(_items(state.get("pickups")), px, py)[:RICH_MAX_PICKUPS]:
            identity = _hash_pair(pickup.get("id"), pickup.get("type"))
            category = str(pickup.get("category", ""))
            one_hot = [1.0 if category == value else 0.0 for value in categories]
            output[cursor:cursor + 11] = (
                *one_hot,
                identity[0], identity[1],
                np.clip(_number(pickup.get("healing")) / max_health, 0.0, 1.0),
                np.clip(_number(pickup.get("material_value")) / 100.0, 0.0, 1.0),
                np.clip(_number(pickup.get("crate_value")), 0.0, 1.0),
                np.clip(_number(pickup.get("width"), 40.0) / width, 0.0, 1.0),
                np.clip(_number(pickup.get("height"), 40.0) / height, 0.0, 1.0),
            )
            cursor += 11

        cursor = RICH_OBSERVATION_SIZE + RICH_MAX_ENEMIES * 10 + RICH_MAX_PICKUPS * 11
        combat = _mapping(state.get("combat"))
        for weapon in _items(combat.get("weapons"))[:SEMANTIC_MAX_WEAPONS]:
            identity = _hash_pair(weapon.get("id"))
            attack_hash = _hash_pair(weapon.get("attack_type"))[0]
            cooldown_duration = max(1e-6, _number(weapon.get("cooldown_duration"), 1.0))
            ammo = _number(weapon.get("ammo"), -1.0)
            capacity = _number(weapon.get("ammo_capacity"), -1.0)
            output[cursor:cursor + 10] = (
                identity[0], identity[1], attack_hash,
                np.clip(_number(weapon.get("range")) / 1000.0, 0.0, 1.0),
                np.clip(_number(weapon.get("cooldown_remaining")) / cooldown_duration, 0.0, 1.0),
                np.clip(_number(weapon.get("reload_remaining")) / 5.0, 0.0, 1.0),
                np.clip(ammo / max(1.0, capacity), 0.0, 1.0) if ammo >= 0.0 else -1.0,
                1.0 if weapon.get("ready") else 0.0,
                1.0 if weapon.get("is_attacking") else 0.0,
                1.0 if weapon.get("is_reloading") else 0.0,
            )
            cursor += 10

        cursor = (
            RICH_OBSERVATION_SIZE
            + RICH_MAX_ENEMIES * 10
            + RICH_MAX_PICKUPS * 11
            + SEMANTIC_MAX_WEAPONS * 10
        )
        enemies = _items(state.get("enemies"))
        enemies_by_runtime = _enemy_runtime_index(enemies)
        indicators = _nearest(_items(state.get("attack_indicators")), px, py)
        for indicator in indicators[:SEMANTIC_MAX_INDICATORS]:
            owner_code = _owner_threat_code(indicator, enemies_by_runtime)
            dx, dy = _relative(indicator, px, py)
            direction_x, direction_y = _xy(indicator.get("direction"))
            output[cursor:cursor + 10] = (
                owner_code,
                np.clip(dx / width, -1.0, 1.0),
                np.clip(dy / height, -1.0, 1.0),
                np.clip(direction_x, -1.0, 1.0),
                np.clip(direction_y, -1.0, 1.0),
                np.clip(_number(indicator.get("width"), 80.0) / width, 0.0, 1.0),
                np.clip(_number(indicator.get("height"), 80.0) / height, 0.0, 1.0),
                np.clip(_number(indicator.get("time_to_activate")) / 5.0, 0.0, 1.0),
                np.clip(_number(indicator.get("damage")) / max_health, 0.0, 1.0),
                1.0 if indicator.get("active") else 0.0,
            )
            cursor += 10
        return output


class FullArenaCombatVectorizer:
    """Whole-arena V4 observation preserving the semantic feature contract.

    The first 832 values remain byte-for-byte compatible with the semantic
    policy.  A coarse spatial map summarizes every entity exported by the API,
    while the nearest enemies receive exact visible attack geometry.
    """

    observation_size = FULL_ARENA_OBSERVATION_SIZE

    def __init__(self):
        self.base = SemanticCombatVectorizer()

    @staticmethod
    def _cell(position: Any, width: float, height: float) -> int:
        x, y = _xy(position)
        column = min(
            FULL_ARENA_GRID_COLUMNS - 1,
            max(0, int(x / width * FULL_ARENA_GRID_COLUMNS)),
        )
        row = min(
            FULL_ARENA_GRID_ROWS - 1,
            max(0, int(y / height * FULL_ARENA_GRID_ROWS)),
        )
        return row * FULL_ARENA_GRID_COLUMNS + column

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray:
        output = np.zeros(FULL_ARENA_OBSERVATION_SIZE, dtype=np.float32)
        output[:SEMANTIC_OBSERVATION_SIZE] = self.base.build(state, previous_action)
        arena = _mapping(state.get("arena"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        player = _mapping(state.get("player"))
        px, py = _xy(player.get("position"))
        max_health = max(1.0, _number(player.get("max_health"), 1.0))
        grid = output[
            SEMANTIC_OBSERVATION_SIZE:
            SEMANTIC_OBSERVATION_SIZE + FULL_ARENA_GRID_SIZE
        ].reshape(-1, FULL_ARENA_GRID_CHANNELS)

        exported_grid = _mapping(state.get("arena_grid")).get("enemy")
        has_exported_grid = (
            isinstance(exported_grid, Iterable)
            and not isinstance(exported_grid, (str, bytes, Mapping))
        )
        if has_exported_grid:
            exported_values = np.asarray(list(exported_grid), dtype=np.float32)
            has_exported_grid = exported_values.shape == (
                FULL_ARENA_GRID_COLUMNS * FULL_ARENA_GRID_ROWS * 4,
            ) and bool(np.isfinite(exported_values).all())
        if has_exported_grid:
            grid[:, :4] = exported_values.reshape(-1, 4)
        else:
            for enemy in _items(state.get("enemies")):
                cell = self._cell(enemy.get("position"), width, height)
                vx, vy = _xy(enemy.get("velocity"))
                radius = max(1.0, _number(enemy.get("radius"), 40.0))
                damage = max(0.0, _number(enemy.get("contact_damage")))
                grid[cell, 0] += 1.0 / 16.0
                grid[cell, 1] += min(1.0, radius / 300.0 + damage / max_health) / 8.0
                grid[cell, 2] += np.clip(vx / 1000.0, -1.0, 1.0) / 8.0
                grid[cell, 3] += np.clip(vy / 1000.0, -1.0, 1.0) / 8.0

        for projectile in _items(state.get("projectiles")):
            cell = self._cell(projectile.get("position"), width, height)
            vx, vy = _xy(projectile.get("velocity"))
            damage = max(0.0, _number(projectile.get("damage")))
            grid[cell, 4] += 1.0 / 16.0
            grid[cell, 5] += min(1.0, damage / max_health) / 8.0
            grid[cell, 6] += np.clip(vx / 1000.0, -1.0, 1.0) / 8.0
            grid[cell, 7] += np.clip(vy / 1000.0, -1.0, 1.0) / 8.0

        for pickup in _items(state.get("pickups")):
            cell = self._cell(pickup.get("position"), width, height)
            category = str(pickup.get("category", pickup.get("kind", ""))).lower()
            if category in {"healing", "consumable"}:
                grid[cell, 8] += max(
                    1.0 / 8.0,
                    min(1.0, _number(pickup.get("healing")) / max_health),
                )
            else:
                value = max(
                    1.0,
                    _number(pickup.get("material_value")),
                    _number(pickup.get("crate_value")) * 10.0,
                )
                grid[cell, 9] += min(1.0, value / 20.0)
        np.clip(grid, -1.0, 1.0, out=grid)

        cursor = SEMANTIC_OBSERVATION_SIZE + FULL_ARENA_GRID_SIZE
        for enemy in _nearest(_items(state.get("enemies")), px, py)[:RICH_MAX_ENEMIES]:
            charge_x, charge_y = _xy(enemy.get("charge_direction"))
            target_x, target_y = _xy(enemy.get("attack_target"))
            target_visible = bool(
                enemy.get("is_attacking")
                or enemy.get("is_charging")
                or target_x
                or target_y
            )
            output[cursor:cursor + FULL_ARENA_ATTACK_FEATURES] = (
                np.clip(charge_x, -1.0, 1.0),
                np.clip(charge_y, -1.0, 1.0),
                np.clip((target_x - px) / width, -1.0, 1.0) if target_visible else 0.0,
                np.clip((target_y - py) / height, -1.0, 1.0) if target_visible else 0.0,
            )
            cursor += FULL_ARENA_ATTACK_FEATURES
        return output


class BulletHellCombatVectorizer:
    """V4 observation with all-projectile future paths and action risks.

    The first 1,512 values are exactly the full-arena generation. Bridge 0.3.8
    appends a player-centered 20x12 map with five time horizons, four separate
    direction lanes, damage weighting, and collision risk for every action.
    """

    observation_size = BULLET_HELL_OBSERVATION_SIZE
    # Risk shaping must be large enough to compete with kill shaping, while
    # remaining smaller than a wave-clear or terminal outcome.
    path_risk_reward_scale = 0.10
    boundary_risk_reward_scale = 0.08
    idle_reward_scale = 0.01
    reversal_reward_scale = 0.004
    low_motion_reward_scale = 0.006

    def __init__(self):
        self.base = FullArenaCombatVectorizer()

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray:
        output = np.zeros(BULLET_HELL_OBSERVATION_SIZE, dtype=np.float32)
        output[:FULL_ARENA_OBSERVATION_SIZE] = self.base.build(state, previous_action)
        cursor = FULL_ARENA_OBSERVATION_SIZE
        paths = _mapping(state.get("projectile_paths"))
        raw_grid = paths.get("grid")
        if (
            isinstance(raw_grid, Iterable)
            and not isinstance(raw_grid, (str, bytes, Mapping))
        ):
            values = np.asarray(list(raw_grid), dtype=np.float32)
            if values.shape == (BULLET_HELL_GRID_SIZE,) and np.isfinite(values).all():
                output[cursor:cursor + BULLET_HELL_GRID_SIZE] = np.clip(
                    values, 0.0, 1.0
                )
        cursor += BULLET_HELL_GRID_SIZE
        for key, size in (
            ("action_risk", BULLET_HELL_PROJECTILE_RISK_SIZE),
            ("enemy_action_risk", BULLET_HELL_ENEMY_RISK_SIZE),
            ("boundary_action_risk", BULLET_HELL_BOUNDARY_RISK_SIZE),
        ):
            raw_risk = paths.get(key)
            if (
                isinstance(raw_risk, Iterable)
                and not isinstance(raw_risk, (str, bytes, Mapping))
            ):
                values = np.asarray(list(raw_risk), dtype=np.float32)
                if values.shape == (size,) and np.isfinite(values).all():
                    output[cursor:cursor + size] = np.clip(values, 0.0, 1.0)
            cursor += size
        output[cursor] = np.clip(
            _number(paths.get("count"), len(_items(state.get("projectiles")))) / 512.0,
            0.0,
            1.0,
        )
        output[cursor + 1] = np.clip(
            _number(paths.get("enemy_count"), len(_items(state.get("enemies")))) / 512.0,
            0.0,
            1.0,
        )
        return output


class SemanticCombatPolicyBase(nn.Module):
    """Small residual actor that preserves the established base behavior."""

    def __init__(self, base: CombatPolicyBase | None = None):
        super().__init__()
        self.base = base or CombatPolicyBase()
        semantic_size = SEMANTIC_OBSERVATION_SIZE - RICH_OBSERVATION_SIZE
        self.semantic = nn.Sequential(
            nn.LayerNorm(semantic_size),
            nn.Linear(semantic_size, 64), nn.Tanh(),
            nn.Linear(64, len(MoveAction)),
        )
        nn.init.zeros_(self.semantic[-1].weight)
        nn.init.zeros_(self.semantic[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        old_logits = self.base(observation[..., :RICH_OBSERVATION_SIZE])
        semantic = observation[..., RICH_OBSERVATION_SIZE:]
        return old_logits + self.semantic(semantic)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CombatPolicyBase(nn.Module):
    """Small behavior-cloning base for future combat fine-tuning."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(RICH_OBSERVATION_SIZE),
            nn.Linear(RICH_OBSERVATION_SIZE, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, len(MoveAction)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_combat_base(path: Path) -> tuple[CombatPolicyBase, dict]:
    """Load the compact human combat base without coupling callers to a runner."""

    resolved = Path(path).resolve()
    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "brotato_combat_base_v1":
        raise RuntimeError(f"unsupported combat BC checkpoint: {resolved}")
    model = CombatPolicyBase()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_semantic_combat_base(path: Path) -> tuple[SemanticCombatPolicyBase, dict]:
    resolved = Path(path).resolve()
    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "brotato_semantic_combat_base_v2":
        raise RuntimeError(f"unsupported semantic combat checkpoint: {resolved}")
    model = SemanticCombatPolicyBase()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


class CombatDecisionLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self.vectorizer = RichCombatVectorizer()

    def record(
        self,
        state: Mapping[str, Any],
        decision: SafetyDecision,
        *,
        source: str,
        previous_action: int,
    ) -> None:
        if self.path is None:
            return
        features = self.vectorizer.build(state, previous_action)
        record = {
            "schema": 1,
            "timestamp": time.time(),
            "source": source,
            "wave": int(_number(_mapping(state.get("wave")).get("number"))),
            "features": [round(float(value), 6) for value in features],
            "requested_action": decision.requested_action,
            "action": decision.applied_action,
            "shielded": decision.overridden,
            "requested_risk": round(decision.requested_risk, 6),
            "applied_risk": round(decision.applied_risk, 6),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class HumanCombatDecisionLogger:
    """Append human movement demonstrations in the combat-BC schema."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.vectorizer = RichCombatVectorizer()

    def record(
        self,
        state: Mapping[str, Any],
        action: int,
        *,
        previous_action: int,
        episode: int,
    ) -> None:
        normalized = int(MoveAction(int(action)))
        features = self.vectorizer.build(state, previous_action)
        record = {
            "schema": 1,
            "dataset": "human_combat_v1",
            "timestamp": time.time(),
            "session": str(state.get("session", "")),
            "episode": int(episode),
            "tick": int(_number(state.get("tick"), -1)),
            "wave": int(_number(_mapping(state.get("wave")).get("number"))),
            "features": [round(float(value), 6) for value in features],
            "previous_action": int(MoveAction(int(previous_action))),
            "action": normalized,
            "human_input_age_ms": int(_number(state.get("human_input_age_ms"), -1)),
            "source": "human_wasd",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class SemanticHumanCombatDecisionLogger:
    """Record API-semantic observations with human movement labels."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.vectorizer = SemanticCombatVectorizer()

    def record(
        self,
        state: Mapping[str, Any],
        action: int,
        *,
        previous_action: int,
        episode: int,
    ) -> None:
        normalized = int(MoveAction(int(action)))
        features = self.vectorizer.build(state, previous_action)
        record = {
            "schema": 2,
            "dataset": "human_semantic_combat_v2",
            "timestamp": time.time(),
            "session": str(state.get("session", "")),
            "episode": int(episode),
            "tick": int(_number(state.get("tick"), -1)),
            "wave": int(_number(_mapping(state.get("wave")).get("number"))),
            "features": [round(float(value), 6) for value in features],
            "previous_action": int(MoveAction(int(previous_action))),
            "action": normalized,
            "human_input_age_ms": int(_number(state.get("human_input_age_ms"), -1)),
            "source": "human_wasd_semantic_api",
            "counts": {
                "enemies": len(_items(state.get("enemies"))),
                "pickups": len(_items(state.get("pickups"))),
                "indicators": len(_items(state.get("attack_indicators"))),
                "weapons": len(_items(_mapping(state.get("combat")).get("weapons"))),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
