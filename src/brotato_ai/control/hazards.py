"""The only runtime hazard scorer."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction
from brotato_ai.domain.decisions import HazardRisk, SafetyDecision
from brotato_ai.domain.state import StateSnapshot

_ACTION_XY = np.array(
    [ACTION_VECTORS[action] for action in MoveAction], dtype=np.float64
)
_METHOD_BUFFER = {
    "charge": 240.0,
    "area": 190.0,
    "contact": 120.0,
    "summon": 110.0,
    "projectile": 90.0,
}


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


def _relative(item: Mapping[str, Any], px: float, py: float) -> tuple[float, float]:
    x, y = _xy(item.get("position"))
    return x - px, y - py


def _nearest(
    items: list[Mapping[str, Any]], px: float, py: float
) -> list[Mapping[str, Any]]:
    return sorted(items, key=lambda item: sum(v * v for v in _relative(item, px, py)))


def _enemy_is_boss(enemy: Mapping[str, Any]) -> bool:
    token = f"{enemy.get('id', '')} {enemy.get('type', '')} {enemy.get('name', '')}".lower()
    return bool(enemy.get("is_boss")) or any(
        term in token for term in ("boss", "summoner", "\u53ec\u5524")
    )


def _enemy_runtime_index(
    enemies: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(enemy.get("runtime_id")): enemy
        for enemy in enemies
        if str(enemy.get("runtime_id", ""))
    }


def _owner_threat_code(
    attack: Mapping[str, Any],
    enemies_by_runtime: Mapping[str, Mapping[str, Any]],
) -> float:
    owner = enemies_by_runtime.get(str(attack.get("owner_runtime_id", "")))
    if owner is None:
        return 0.0
    return 1.0 if _enemy_is_boss(owner) else 0.5


def enemy_separation_diagnostics(
    state: Mapping[str, Any] | StateSnapshot,
    action: int,
    *,
    horizon_seconds: float = 0.45,
) -> dict[str, float | bool | str]:
    """Shared moving-enemy separation estimate for scoring and tactical control."""

    empty: dict[str, float | bool | str] = {
        "active": False,
        "ranged_active": False,
        "current_distance": 0.0,
        "predicted_distance": 0.0,
        "target_distance": 0.0,
        "closing_rate": 0.0,
        "radial_dot": 0.0,
        "enemy_runtime_id": "",
    }
    payload = state.payload if isinstance(state, StateSnapshot) else state
    combat = _mapping(payload.get("combat"))
    ranged_count = _number(combat.get("ranged_count"), 0.0)
    melee_count = _number(combat.get("melee_count"), 0.0)
    weapon_range = _number(combat.get("weapon_range"), 0.0)
    ranged_active = ranged_count > 0.0 and ranged_count > melee_count and weapon_range > 0.0
    player = _mapping(payload.get("player"))
    px, py = _xy(player.get("position"))
    player_radius = max(
        18.0,
        _number(player.get("radius"), 28.0),
        _number(player.get("width"), 0.0) * 0.5,
        _number(player.get("height"), 0.0) * 0.5,
    )
    speed = max(150.0, _number(combat.get("move_speed"), 300.0))
    movement = ACTION_VECTORS[MoveAction(int(action))]
    future_x = px + movement[0] * speed * horizon_seconds
    future_y = py + movement[1] * speed * horizon_seconds
    candidates: list[tuple[float, float, float, float, Mapping[str, Any]]] = []
    for enemy in _items(payload.get("enemies")):
        if bool(enemy.get("dead")):
            continue
        ex, ey = _xy(enemy.get("position"))
        evx, evy = _xy(enemy.get("velocity"))
        predicted_distance = math.hypot(
            ex + evx * horizon_seconds - future_x,
            ey + evy * horizon_seconds - future_y,
        )
        current_distance = math.hypot(ex - px, ey - py)
        enemy_radius = max(25.0, _number(enemy.get("radius"), 40.0))
        contact_clearance = player_radius + enemy_radius + 80.0
        target_distance = contact_clearance
        if ranged_active:
            target_distance = max(
                contact_clearance,
                min(420.0, max(180.0, weapon_range * 0.55)),
            )
        candidates.append(
            (
                predicted_distance,
                current_distance,
                target_distance,
                max(0.0, current_distance - predicted_distance),
                enemy,
            )
        )
    if not candidates:
        empty["ranged_active"] = ranged_active
        return empty
    predicted_distance, current_distance, target_distance, closing_distance, enemy = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    ex, ey = _xy(enemy.get("position"))
    away_x, away_y = px - ex, py - ey
    away_length = max(1.0, math.hypot(away_x, away_y))
    radial_dot = (
        movement[0] * away_x / away_length
        + movement[1] * away_y / away_length
    )
    closing_rate = closing_distance / max(1.0, target_distance)
    if predicted_distance > target_distance * 1.40:
        closing_rate = 0.0
    return {
        "active": True,
        "ranged_active": ranged_active,
        "current_distance": float(current_distance),
        "predicted_distance": float(predicted_distance),
        "target_distance": float(target_distance),
        "closing_rate": float(closing_rate),
        "radial_dot": float(radial_dot),
        "enemy_runtime_id": str(enemy.get("runtime_id", "")),
    }


def ranged_spacing_diagnostics(
    state: Mapping[str, Any] | StateSnapshot,
    action: int,
    *,
    horizon_seconds: float = 0.45,
    enabled: bool = True,
) -> dict[str, float | bool]:
    """Return transparent ranged-spacing diagnostics for one candidate action."""

    empty = {
        "active": False,
        "risk": 0.0,
        "target_distance": 0.0,
        "predicted_distance": 0.0,
        "closing_rate": 0.0,
        "spacing_error": 0.0,
    }
    if not enabled:
        return empty
    diagnostic = enemy_separation_diagnostics(
        state, action, horizon_seconds=horizon_seconds
    )
    if not diagnostic["active"] or not diagnostic["ranged_active"]:
        return empty
    target_distance = float(diagnostic["target_distance"])
    predicted_distance = float(diagnostic["predicted_distance"])
    spacing_error = max(0.0, target_distance - predicted_distance) / max(1.0, target_distance)
    closing_rate = float(diagnostic["closing_rate"])
    risk = min(
        2.0,
        1.25 * spacing_error * spacing_error
        + 0.35 * min(1.0, closing_rate),
    )
    return {
        "active": True,
        "risk": float(risk),
        "target_distance": target_distance,
        "predicted_distance": predicted_distance,
        "closing_rate": closing_rate,
        "spacing_error": float(spacing_error),
    }


def ranged_spacing_risk(
    state: Mapping[str, Any] | StateSnapshot,
    action: int,
    *,
    horizon_seconds: float = 0.45,
    enabled: bool = True,
) -> float:
    return float(
        ranged_spacing_diagnostics(
            state,
            action,
            horizon_seconds=horizon_seconds,
            enabled=enabled,
        )["risk"]
    )


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


CROWD_RADIUS = 240.0


def _crowd_density_risk(
    enemies: list[Mapping[str, Any]],
    future_x: float,
    future_y: float,
    movement: tuple[float, float],
    px: float,
    py: float,
) -> float:
    """Risk from a surround that never enters a single-enemy danger radius."""

    nearby_x: list[float] = []
    nearby_y: list[float] = []
    for enemy in enemies:
        if bool(enemy.get("dead")):
            continue
        ex, ey = _xy(enemy.get("position"))
        evx, evy = _xy(enemy.get("velocity"))
        predicted_x = ex + evx * 0.45
        predicted_y = ey + evy * 0.45
        if math.hypot(predicted_x - future_x, predicted_y - future_y) >= CROWD_RADIUS:
            continue
        nearby_x.append(predicted_x)
        nearby_y.append(predicted_y)
    if len(nearby_x) <= 1:
        return 0.0
    density = min(4.0, (len(nearby_x) / 6.0) ** 2 * 0.9)
    mx = sum(nearby_x) / len(nearby_x) - px
    my = sum(nearby_y) / len(nearby_y) - py
    length = max(1.0, math.hypot(mx, my))
    toward = max(0.0, movement[0] * mx / length + movement[1] * my / length)
    return float(density * (1.0 + 0.6 * toward))


class UnifiedHazardScorer:
    """Score enemy, projectile, telegraph, and boundary risk for every action."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        override_margin: float = 0.08,
        hard_risk_threshold: float = 0.65,
        minimum_risk: float = 0.22,
        switch_penalty: float = 0.05,
        ranged_spacing_enabled: bool = True,
        ranged_spacing_weight: float = 1.25,
    ):
        self.enabled = bool(enabled)
        self.override_margin = float(override_margin)
        self.hard_risk_threshold = float(hard_risk_threshold)
        self.minimum_risk = max(0.0, float(minimum_risk))
        self.switch_penalty = max(0.0, float(switch_penalty))
        self.ranged_spacing_enabled = bool(ranged_spacing_enabled)
        self.ranged_spacing_weight = max(0.0, float(ranged_spacing_weight))

    @staticmethod
    def _payload(state: Mapping[str, Any] | StateSnapshot) -> Mapping[str, Any]:
        return state.payload if isinstance(state, StateSnapshot) else state

    def risk_breakdown(
        self, state: Mapping[str, Any] | StateSnapshot, action: int
    ) -> HazardRisk:
        state = self._payload(state)
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
        spacing = ranged_spacing_diagnostics(
            state,
            action,
            enabled=self.ranged_spacing_enabled,
        )
        spacing_risk = float(spacing["risk"]) * self.ranged_spacing_weight

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
        enemy_risk += _crowd_density_risk(
            enemies, future_x, future_y, movement, px, py
        )

        wave_number = int(_number(_mapping(state.get("wave")).get("number"), 0.0))
        boss_present = any(_enemy_is_boss(enemy) for enemy in enemies)
        if not boss_present and wave_number >= 20:
            boss_present = bool(
                _items(state.get("attack_indicators"))
                or _items(state.get("projectiles"))
            )
        if boss_present:
            player_radius = max(18.0, _number(player.get("radius"), 28.0))
            for enemy in enemies:
                if not _enemy_is_boss(enemy):
                    continue
                ex, ey = _xy(enemy.get("position"))
                boss_distance = math.hypot(ex - future_x, ey - future_y)
                boss_radius = max(45.0, _number(enemy.get("radius"), 55.0))
                separation = max(480.0, boss_radius + player_radius + 300.0)
                if boss_distance < separation:
                    enemy_risk += ((separation - boss_distance) / separation) ** 2 * 10.0

        for projectile in _nearest(_items(state.get("projectiles")), px, py)[:32]:
            if "hostile" in projectile and not bool(projectile.get("hostile")):
                continue
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

        for indicator in _nearest(_items(state.get("attack_indicators")), px, py)[:32]:
            token = f"{indicator.get('id', '')} {indicator.get('type', '')}".lower()
            owner_is_boss = _owner_threat_code(indicator, enemies_by_runtime) >= 1.0
            if (
                not owner_is_boss
                and not boss_present
                and not any(term in token for term in ("aoe", "warning", "circle", "boss"))
            ):
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
                indicator_risk += (
                    (18.0 if imminent else 9.0)
                    if owner_is_boss
                    else (14.0 if imminent else 7.0)
                )
            else:
                gap_x = max(0.0, dx - half_width)
                gap_y = max(0.0, dy - half_height)
                distance = math.hypot(gap_x, gap_y)
                if distance < 180.0:
                    indicator_risk += (1.0 - distance / 180.0) ** 2 * (
                        5.0 if imminent else 2.0
                    )

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
            ranged_spacing=float(spacing_risk),
        )

    def risk(self, state: Mapping[str, Any] | StateSnapshot, action: int) -> float:
        return self.risk_breakdown(state, action).total

    def all_risks(
        self, state: Mapping[str, Any] | StateSnapshot
    ) -> dict[int, HazardRisk]:
        payload = self._payload(state)
        return self._all_risks_vectorized(payload)

    def _all_risks_vectorized(
        self, payload: Mapping[str, Any]
    ) -> dict[int, HazardRisk]:
        """Score nine actions with packed arrays. Same math as risk_breakdown."""

        player = _mapping(payload.get("player"))
        px, py = _xy(player.get("position"))
        arena = _mapping(payload.get("arena"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        combat = _mapping(payload.get("combat"))
        player_speed = max(150.0, _number(combat.get("move_speed"), 300.0))
        future = np.array([px, py], dtype=np.float64) + _ACTION_XY * player_speed * 0.45
        enemies = _items(payload.get("enemies"))
        projectiles = _items(payload.get("projectiles"))
        indicators = _items(payload.get("attack_indicators"))
        enemies_by_runtime = _enemy_runtime_index(enemies)
        ranked_enemies = _nearest(enemies, px, py)[:24]
        ranked_projectiles = _nearest(projectiles, px, py)[:32]
        ranked_indicators = _nearest(indicators, px, py)[:32]
        wave_number = int(_number(_mapping(payload.get("wave")).get("number"), 0.0))
        boss_present = any(_enemy_is_boss(enemy) for enemy in enemies)
        if not boss_present and wave_number >= 20:
            boss_present = bool(indicators or projectiles)
        player_radius = max(18.0, _number(player.get("radius"), 28.0))

        edge_distance = np.minimum.reduce(
            [future[:, 0], width - future[:, 0], future[:, 1], height - future[:, 1]]
        )
        boundary_risk = np.zeros(len(MoveAction), dtype=np.float64)
        edge_hit = edge_distance < 150.0
        if np.any(edge_hit):
            gap = (150.0 - edge_distance[edge_hit]) / 150.0
            boundary_risk[edge_hit] = gap * gap * 4.0

        enemy_risk = np.zeros(len(MoveAction), dtype=np.float64)
        if ranked_enemies:
            predicted_x = np.array(
                [
                    _xy(enemy.get("position"))[0] + _xy(enemy.get("velocity"))[0] * 0.45
                    for enemy in ranked_enemies
                ],
                dtype=np.float64,
            )
            predicted_y = np.array(
                [
                    _xy(enemy.get("position"))[1] + _xy(enemy.get("velocity"))[1] * 0.45
                    for enemy in ranked_enemies
                ],
                dtype=np.float64,
            )
            methods = [
                str(enemy.get("attack_method", "unknown")).lower()
                for enemy in ranked_enemies
            ]
            danger = np.array(
                [
                    max(25.0, _number(enemy.get("radius"), 40.0))
                    + _METHOD_BUFFER.get(method, 100.0)
                    + (60.0 if _enemy_is_boss(enemy) else 0.0)
                    for enemy, method in zip(ranked_enemies, methods)
                ],
                dtype=np.float64,
            )
            weight = np.array(
                [
                    6.0
                    if bool(enemy.get("is_charging")) or method == "charge"
                    else 2.0
                    for enemy, method in zip(ranked_enemies, methods)
                ],
                dtype=np.float64,
            )
            distance = np.hypot(
                predicted_x[None, :] - future[:, 0:1],
                predicted_y[None, :] - future[:, 1:2],
            )
            inside = distance < danger[None, :]
            if np.any(inside):
                norm = np.where(inside, (danger[None, :] - distance) / danger[None, :], 0.0)
                enemy_risk += np.sum(norm * norm * weight[None, :], axis=1)
        for action in MoveAction:
            movement = ACTION_VECTORS[MoveAction(int(action))]
            enemy_risk[int(action)] += _crowd_density_risk(
                enemies,
                float(future[int(action), 0]),
                float(future[int(action), 1]),
                movement,
                px,
                py,
            )

        if boss_present:
            for enemy in enemies:
                if not _enemy_is_boss(enemy):
                    continue
                ex, ey = _xy(enemy.get("position"))
                boss_radius = max(45.0, _number(enemy.get("radius"), 55.0))
                separation = max(480.0, boss_radius + player_radius + 300.0)
                boss_distance = np.hypot(ex - future[:, 0], ey - future[:, 1])
                close = boss_distance < separation
                if np.any(close):
                    norm = (separation - boss_distance[close]) / separation
                    enemy_risk[close] += norm * norm * 10.0

        projectile_risk = np.zeros(len(MoveAction), dtype=np.float64)
        near_projectiles = [
            projectile
            for projectile in ranked_projectiles
            if not ("hostile" in projectile and not bool(projectile.get("hostile")))
        ]
        if near_projectiles:
            qx = np.array([_xy(item.get("position"))[0] for item in near_projectiles], dtype=np.float64)
            qy = np.array([_xy(item.get("position"))[1] for item in near_projectiles], dtype=np.float64)
            qvx = np.array([_xy(item.get("velocity"))[0] for item in near_projectiles], dtype=np.float64)
            qvy = np.array([_xy(item.get("velocity"))[1] for item in near_projectiles], dtype=np.float64)
            radius = np.array(
                [max(8.0, _number(item.get("radius"), 12.0)) + 42.0 for item in near_projectiles],
                dtype=np.float64,
            )
            stationary = qvx * qvx + qvy * qvy < 100.0
            danger = radius * np.where(stationary, 2.3, 1.5)
            owner_is_boss = np.array(
                [_owner_threat_code(item, enemies_by_runtime) >= 1.0 for item in near_projectiles],
                dtype=bool,
            )
            rx = qx - px
            ry = qy - py
            rvx = qvx[None, :] - _ACTION_XY[:, 0:1] * player_speed
            rvy = qvy[None, :] - _ACTION_XY[:, 1:2] * player_speed
            speed_sq = rvx * rvx + rvy * rvy
            closest_time = np.zeros_like(speed_sq)
            moving = speed_sq >= 1.0
            closest_time[moving] = np.clip(
                -(rx[None, :] * rvx + ry[None, :] * rvy)[moving] / speed_sq[moving],
                0.0,
                0.8,
            )
            miss = np.hypot(
                rx[None, :] + rvx * closest_time, ry[None, :] + rvy * closest_time
            )
            hit = miss < danger[None, :]
            if np.any(hit):
                urgency = np.where(
                    stationary[None, :], 1.0, 1.0 - np.minimum(1.0, closest_time / 0.8)
                )
                scale = np.where(
                    owner_is_boss[None, :],
                    7.0 + 10.0 * urgency,
                    np.where(boss_present, 6.0 + 8.0 * urgency, 3.0 + 5.0 * urgency),
                )
                norm = np.where(hit, (danger[None, :] - miss) / danger[None, :], 0.0)
                projectile_risk += np.sum(norm * norm * scale, axis=1)

        indicator_risk = np.zeros(len(MoveAction), dtype=np.float64)
        for indicator in ranked_indicators:
            token = f"{indicator.get('id', '')} {indicator.get('type', '')}".lower()
            owner_is_boss = _owner_threat_code(indicator, enemies_by_runtime) >= 1.0
            if (
                not owner_is_boss
                and not boss_present
                and not any(term in token for term in ("aoe", "warning", "circle", "boss"))
            ):
                continue
            ix, iy = _xy(indicator.get("position"))
            half_width = max(35.0, _number(indicator.get("width"), 80.0) * 0.5) + 45.0
            half_height = max(35.0, _number(indicator.get("height"), 80.0) * 0.5) + 45.0
            dx = np.abs(future[:, 0] - ix)
            dy = np.abs(future[:, 1] - iy)
            inside = (dx <= half_width) & (dy <= half_height)
            time_to_activate = max(0.0, _number(indicator.get("time_to_activate"), 5.0))
            imminent = bool(indicator.get("active")) or time_to_activate <= 1.25
            inside_score = (
                (18.0 if imminent else 9.0) if owner_is_boss else (14.0 if imminent else 7.0)
            )
            indicator_risk[inside] += inside_score
            outside = ~inside
            if np.any(outside):
                gap_x = np.maximum(0.0, dx[outside] - half_width)
                gap_y = np.maximum(0.0, dy[outside] - half_height)
                distance = np.hypot(gap_x, gap_y)
                near = distance < 180.0
                if np.any(near):
                    indicator_risk[np.flatnonzero(outside)[near]] += (
                        (1.0 - distance[near] / 180.0) ** 2
                    ) * (5.0 if imminent else 2.0)

        paths = _mapping(payload.get("projectile_paths"))
        projectile_path_risk = np.zeros(len(MoveAction), dtype=np.float64)
        enemy_path_risk = np.zeros(len(MoveAction), dtype=np.float64)
        boundary_path_risk = np.zeros(len(MoveAction), dtype=np.float64)
        for key, scale, target in (
            ("action_risk", 1.5, projectile_path_risk),
            ("enemy_action_risk", 1.5, enemy_path_risk),
            ("boundary_action_risk", 1.25, boundary_path_risk),
        ):
            values = paths.get(key, [])
            if not isinstance(values, (list, tuple)):
                continue
            for action in MoveAction:
                if int(action) >= len(values):
                    continue
                target[int(action)] += max(0.0, min(1.0, _number(values[int(action)]))) * scale

        spacing_risk = np.zeros(len(MoveAction), dtype=np.float64)
        if self.ranged_spacing_enabled:
            for action in MoveAction:
                spacing_risk[int(action)] = (
                    float(
                        ranged_spacing_diagnostics(
                            payload, int(action), enabled=True
                        )["risk"]
                    )
                    * self.ranged_spacing_weight
                )

        return {
            int(action): HazardRisk(
                enemy=float(enemy_risk[int(action)]),
                projectile=float(projectile_risk[int(action)]),
                indicator=float(indicator_risk[int(action)]),
                boundary=float(boundary_risk[int(action)]),
                enemy_path=float(enemy_path_risk[int(action)]),
                projectile_path=float(projectile_path_risk[int(action)]),
                boundary_path=float(boundary_path_risk[int(action)]),
                ranged_spacing=float(spacing_risk[int(action)]),
            )
            for action in MoveAction
        }

    def apply(
        self,
        state: Mapping[str, Any] | StateSnapshot,
        requested_action: int,
        *,
        previous_action: int | None = None,
    ) -> SafetyDecision:
        if not self.enabled:
            requested = int(MoveAction(int(requested_action)))
            return SafetyDecision(requested, requested, 0.0, 0.0)
        risks = self.all_risks(state)
        return self.choose(risks, requested_action, previous_action=previous_action)

    def choose(
        self,
        risks: Mapping[int, HazardRisk],
        requested_action: int,
        *,
        previous_action: int | None = None,
    ) -> SafetyDecision:
        """Choose from one precomputed nine-action hazard assessment."""

        requested = int(MoveAction(int(requested_action)))
        if not self.enabled:
            return SafetyDecision(requested, requested, 0.0, 0.0)
        requested_risk = risks[requested].total
        raw_best_action = min(
            risks,
            key=lambda action: (
                risks[action].total,
                action == int(MoveAction.IDLE),
                action,
            ),
        )
        raw_best_risk = risks[raw_best_action].total
        if requested_risk - raw_best_risk >= self.override_margin:
            return SafetyDecision(
                requested, raw_best_action, requested_risk, raw_best_risk
            )
        if requested_risk < self.minimum_risk:
            return SafetyDecision(requested, requested, requested_risk, requested_risk)
        previous = (
            int(MoveAction(int(previous_action))) if previous_action is not None else None
        )
        best_action = min(
            risks,
            key=lambda action: (
                risks[action].total
                + (self.switch_penalty if previous is not None and action != previous else 0.0),
                action == int(MoveAction.IDLE),
                action,
            ),
        )
        best_risk = risks[best_action].total
        if (
            requested_risk < self.hard_risk_threshold
            and requested_risk - best_risk < self.override_margin
        ):
            best_action, best_risk = requested, requested_risk
        return SafetyDecision(requested, best_action, requested_risk, best_risk)


CombatSafetyShield = UnifiedHazardScorer
