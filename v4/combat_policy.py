"""Finite-history GRU inputs and macro objectives for V4 movement."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from v3.combat_policy import (
    ACTION_VECTORS,
    BULLET_HELL_OBSERVATION_SIZE,
    BulletHellCombatVectorizer,
    _enemy_is_boss,
    _enemy_runtime_index,
    _owner_threat_code,
)
from v3.protocol import MoveAction


HISTORY_STEPS = 8
HISTORY_FEATURES = 16
HISTORY_SIZE = HISTORY_STEPS * HISTORY_FEATURES
MACRO_OBJECTIVES = 5
MACRO_FEATURES = MACRO_OBJECTIVES + 3
V4_OBSERVATION_SIZE = BULLET_HELL_OBSERVATION_SIZE + HISTORY_SIZE + MACRO_FEATURES

OBJECTIVE_EVADE = 0
OBJECTIVE_HEAL = 1
OBJECTIVE_LOOT = 2
OBJECTIVE_ENGAGE = 3
OBJECTIVE_REPOSITION = 4


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _xy(value: Any) -> tuple[float, float]:
    value = _mapping(value)
    return _number(value.get("x")), _number(value.get("y"))


def _risk_vector(paths: Mapping[str, Any], key: str) -> np.ndarray:
    output = np.zeros(len(MoveAction), dtype=np.float32)
    values = paths.get(key, [])
    if not isinstance(values, (list, tuple)):
        return output
    for index, value in enumerate(values[:len(output)]):
        output[index] = np.clip(_number(value), 0.0, 1.0)
    return output


def _maximum(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _boss_escape_risk(
    state: Mapping[str, Any],
    combined: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    """Add boss bodies and boss-owned telegraphs to per-action path risk."""

    player = _mapping(state.get("player"))
    px, py = _xy(player.get("position"))
    combat = _mapping(state.get("combat"))
    speed = max(150.0, _number(combat.get("move_speed"), 300.0))
    player_radius = max(
        18.0,
        _number(player.get("radius"), 28.0),
        _number(player.get("width"), 0.0) * 0.5,
        _number(player.get("height"), 0.0) * 0.5,
    )
    enemies = _items(state.get("enemies"))
    enemies_by_runtime = _enemy_runtime_index(enemies)
    bosses = [enemy for enemy in enemies if _enemy_is_boss(enemy)]
    indicators = _items(state.get("attack_indicators"))
    projectiles = _items(state.get("projectiles"))
    boss_owned_attacks = sum(
        _owner_threat_code(attack, enemies_by_runtime) >= 1.0
        for attack in (*indicators, *projectiles)
    )
    wave = int(_number(_mapping(state.get("wave")).get("number"), 0.0))
    boss_mode = bool(bosses) or (wave >= 20 and bool(indicators or projectiles))
    if not boss_mode:
        return combined, False, 0.0

    risks = np.asarray(combined, dtype=np.float32).copy()
    proximity = 0.0
    for action, movement in ACTION_VECTORS.items():
        future_x = px + movement[0] * speed * 0.55
        future_y = py + movement[1] * speed * 0.55
        action_risk = float(risks[int(action)])
        for boss in bosses:
            bx, by = _xy(boss.get("position"))
            boss_radius = max(
                45.0,
                _number(boss.get("radius"), 55.0),
                _number(boss.get("width"), 0.0) * 0.5,
                _number(boss.get("height"), 0.0) * 0.5,
            )
            method = str(boss.get("attack_method", "unknown")).lower()
            method_buffer = 360.0 if method in {"charge", "area"} else 300.0
            separation = boss_radius + player_radius + method_buffer
            distance = np.hypot(bx - future_x, by - future_y)
            if distance < separation:
                body_risk = float((separation - distance) / separation)
                action_risk += body_risk
                proximity = max(proximity, body_risk)
        for indicator in indicators:
            if _owner_threat_code(indicator, enemies_by_runtime) < 1.0:
                continue
            ix, iy = _xy(indicator.get("position"))
            half_width = max(35.0, _number(indicator.get("width"), 80.0) * 0.5) + player_radius
            half_height = max(35.0, _number(indicator.get("height"), 80.0) * 0.5) + player_radius
            dx = abs(future_x - ix)
            dy = abs(future_y - iy)
            if dx <= half_width and dy <= half_height:
                imminent = bool(indicator.get("active")) or _number(
                    indicator.get("time_to_activate"), 5.0
                ) <= 1.25
                action_risk += 1.0 if imminent else 0.55
        risks[int(action)] = np.clip(action_risk, 0.0, 1.0)
    owner_urgency = min(1.0, boss_owned_attacks / 6.0)
    return risks, True, max(proximity, owner_urgency)


class HierarchicalCombatVectorizer:
    """V4 observation: V3 prefix, eight transitions, and a macro objective.

    The V3 prefix is byte-for-byte compatible with the trained bullet policy.
    A GRU can therefore learn temporal corrections without discarding the old
    actor. Macro objectives deliberately remain transparent and inspectable.
    """

    observation_size = V4_OBSERVATION_SIZE
    path_risk_reward_scale = 0.10
    boundary_risk_reward_scale = 0.08
    # The previous values were too small relative to the per-step survival
    # reward.  The policy learned to alternate directions near spawn center,
    # producing low displacement without paying a meaningful cost.
    idle_reward_scale = 0.03
    reversal_reward_scale = 0.008
    low_motion_reward_scale = 0.02
    enemy_contact_guard = True
    enemy_contact_guard_threshold = 0.22
    enemy_contact_guard_margin = 0.08
    enemy_contact_override_penalty = 0.02

    def __init__(self):
        self.base = BulletHellCombatVectorizer()
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_STEPS)
        self.previous_snapshot: dict[str, float] | None = None
        self.last_tick: tuple[str, int] | None = None
        self.reset()

    def reset(self, state: Mapping[str, Any] | None = None) -> None:
        self.history.clear()
        for _ in range(HISTORY_STEPS):
            self.history.append(np.zeros(HISTORY_FEATURES, dtype=np.float32))
        self.previous_snapshot = self._snapshot(state or {}) if state else None
        self.last_tick = (
            (str(state.get("session", "")), int(_number(state.get("tick"), -1)))
            if state
            else None
        )

    @staticmethod
    def _snapshot(state: Mapping[str, Any]) -> dict[str, float]:
        player = _mapping(state.get("player"))
        x, y = _xy(player.get("position"))
        health = _number(player.get("health"))
        maximum = max(1.0, _number(player.get("max_health"), 1.0))
        return {"x": x, "y": y, "health": health / maximum}

    def _append_transition(self, state: Mapping[str, Any], previous_action: int) -> None:
        identity = (str(state.get("session", "")), int(_number(state.get("tick"), -1)))
        if identity == self.last_tick:
            return
        self.last_tick = identity
        current = self._snapshot(state)
        previous = self.previous_snapshot
        self.previous_snapshot = current
        if previous is None:
            return
        frame = np.zeros(HISTORY_FEATURES, dtype=np.float32)
        action = int(MoveAction(int(previous_action)))
        frame[action] = 1.0
        frame[9] = np.clip((current["x"] - previous["x"]) / 100.0, -1.0, 1.0)
        frame[10] = np.clip((current["y"] - previous["y"]) / 100.0, -1.0, 1.0)
        frame[11] = np.clip(np.hypot(frame[9], frame[10]), 0.0, 1.0)
        frame[12] = np.clip((current["health"] - previous["health"]) * 10.0, -1.0, 1.0)
        paths = _mapping(state.get("projectile_paths"))
        frame[13] = _maximum(_risk_vector(paths, "action_risk"))
        frame[14] = _maximum(_risk_vector(paths, "enemy_action_risk"))
        frame[15] = _maximum(_risk_vector(paths, "boundary_action_risk"))
        self.history.append(frame)

    @staticmethod
    def _nearest(items: Any, px: float, py: float) -> Mapping[str, Any] | None:
        if not isinstance(items, (list, tuple)):
            return None
        valid = [item for item in items if isinstance(item, Mapping)]
        return min(
            valid,
            key=lambda item: (_xy(item.get("position"))[0] - px) ** 2
            + (_xy(item.get("position"))[1] - py) ** 2,
            default=None,
        )

    def _macro(self, state: Mapping[str, Any]) -> np.ndarray:
        output = np.zeros(MACRO_FEATURES, dtype=np.float32)
        player = _mapping(state.get("player"))
        arena = _mapping(state.get("arena"))
        px, py = _xy(player.get("position"))
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        health = _number(player.get("health")) / max(
            1.0, _number(player.get("max_health"), 1.0)
        )
        paths = _mapping(state.get("projectile_paths"))
        projectile_risk = _risk_vector(paths, "action_risk")
        enemy_risk = _risk_vector(paths, "enemy_action_risk")
        boundary_risk = _risk_vector(paths, "boundary_action_risk")
        combined = np.clip(projectile_risk + enemy_risk + boundary_risk, 0.0, 1.0)
        combined, boss_mode, boss_urgency = _boss_escape_risk(state, combined)
        threat = _maximum(combined)
        pickups = state.get("pickups", [])
        healing = [
            item for item in pickups
            if isinstance(item, Mapping) and _number(item.get("healing")) > 0.0
        ] if isinstance(pickups, (list, tuple)) else []
        target = None
        urgency = 0.0
        if boss_mode or threat >= 0.35:
            objective = OBJECTIVE_EVADE
            safest = int(np.argmin(combined))
            output[-3:-1] = ACTION_VECTORS[MoveAction(safest)]
            urgency = max(threat, boss_urgency)
        elif health < 0.65 and healing:
            objective = OBJECTIVE_HEAL
            target = self._nearest(healing, px, py)
            urgency = 1.0 - health
        elif pickups:
            objective = OBJECTIVE_LOOT
            target = self._nearest(pickups, px, py)
            urgency = 0.5
        elif state.get("enemies"):
            objective = OBJECTIVE_ENGAGE
            target = self._nearest(state.get("enemies"), px, py)
            urgency = 0.35
        else:
            objective = OBJECTIVE_REPOSITION
            # Do not turn the arena center into a permanent attractor.  The
            # safety shield already protects the edges, while the movement
            # reward handles genuine stalling.
            output[-3:-1] = 0.0
            urgency = 0.2
        output[objective] = 1.0
        if target is not None:
            tx, ty = _xy(target.get("position"))
            output[-3] = np.clip((tx - px) / width, -1.0, 1.0)
            output[-2] = np.clip((ty - py) / height, -1.0, 1.0)
        output[-1] = np.clip(urgency, 0.0, 1.0)
        return output

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray:
        self._append_transition(state, previous_action)
        output = np.zeros(V4_OBSERVATION_SIZE, dtype=np.float32)
        output[:BULLET_HELL_OBSERVATION_SIZE] = self.base.build(state, previous_action)
        cursor = BULLET_HELL_OBSERVATION_SIZE
        output[cursor:cursor + HISTORY_SIZE] = np.concatenate(tuple(self.history))
        output[cursor + HISTORY_SIZE:] = self._macro(state)
        return output
