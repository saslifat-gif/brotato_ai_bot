"""Finite-history GRU inputs and macro objectives for V4 movement."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Mapping

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
TRAJECTORY_ENTITY_FEATURES = 8
TRAJECTORY_PROJECTILE_START = 0
TRAJECTORY_ENEMY_START = TRAJECTORY_ENTITY_FEATURES
TRAJECTORY_PROJECTILE_COUNT = TRAJECTORY_ENEMY_START + TRAJECTORY_ENTITY_FEATURES
TRAJECTORY_ENEMY_COUNT = TRAJECTORY_PROJECTILE_COUNT + 1
TRAJECTORY_PROJECTILE_TRACKED = TRAJECTORY_ENEMY_COUNT + 1
TRAJECTORY_ENEMY_TRACKED = TRAJECTORY_PROJECTILE_TRACKED + 1
TRAJECTORY_FEATURES = TRAJECTORY_ENEMY_TRACKED + 1
V4_OBSERVATION_SIZE = (
    BULLET_HELL_OBSERVATION_SIZE
    + HISTORY_SIZE
    + MACRO_FEATURES
    + TRAJECTORY_FEATURES
)

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


def _top_action_threat(values: np.ndarray, count: int = 3) -> float:
    """Estimate global danger without letting one bad direction dominate."""

    if not values.size:
        return 0.0
    top = np.sort(np.asarray(values, dtype=np.float32))[-max(1, min(count, values.size)):]
    return float(np.mean(top))


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


class ThreeFrameTrajectoryTracker:
    """Track short object histories and expose deterministic motion features.

    The bridge already gives entities stable runtime IDs and publishes a real
    timestamp. Three samples are enough for a local velocity and acceleration
    estimate, while the compact output keeps the V4 policy inexpensive.
    """

    horizon_seconds = 0.25
    fallback_dt = 1.0 / 24.0
    stale_after_seconds = 0.75
    projectile_speed_scale = 1200.0
    enemy_speed_scale = 500.0
    acceleration_scale = 5000.0

    def __init__(self) -> None:
        self._tracks: dict[tuple[str, str], dict[str, Any]] = {}
        self._last_identity: tuple[str, int] | None = None
        self._last_timestamp: float | None = None
        self._last_features = np.zeros(TRAJECTORY_FEATURES, dtype=np.float32)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_identity = None
        self._last_timestamp = None
        self._last_features.fill(0.0)

    @classmethod
    def _timestamp(cls, state: Mapping[str, Any], previous: float | None) -> float:
        published_ms = _number(state.get("published_at_ms"), 0.0)
        timestamp = published_ms / 1000.0 if published_ms > 0.0 else 0.0
        if timestamp <= 0.0:
            timestamp = _number(state.get("tick"), 0.0) * cls.fallback_dt
        if previous is not None and timestamp <= previous:
            timestamp = previous + cls.fallback_dt
        return timestamp

    @staticmethod
    def _entity_key(kind: str, entity: Mapping[str, Any]) -> str:
        return str(entity.get("runtime_id") or entity.get("id") or "").strip()

    @staticmethod
    def _reported_velocity(entity: Mapping[str, Any]) -> tuple[float, float]:
        return _xy(entity.get("velocity"))

    @classmethod
    def _motion(
        cls,
        samples: Deque[tuple[float, float, float]],
        entity: Mapping[str, Any],
    ) -> tuple[float, float, float, float, float, float]:
        points = list(samples)
        vx, vy = cls._reported_velocity(entity)
        ax = ay = 0.0
        confidence = 0.0
        if len(points) >= 2:
            _, x0, y0 = points[-2]
            t1, x1, y1 = points[-1]
            dt = max(0.001, t1 - points[-2][0])
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            confidence = 0.5
            if len(points) >= 3:
                t_prev, x_prev, y_prev = points[-3]
                dt_prev = max(0.001, points[-2][0] - t_prev)
                prev_vx = (x0 - x_prev) / dt_prev
                prev_vy = (y0 - y_prev) / dt_prev
                avg_dt = max(0.001, (dt + dt_prev) * 0.5)
                ax = (vx - prev_vx) / avg_dt
                ay = (vy - prev_vy) / avg_dt
                acceleration = float(np.hypot(ax, ay))
                if acceleration > cls.acceleration_scale:
                    scale = cls.acceleration_scale / acceleration
                    ax *= scale
                    ay *= scale
                confidence = 1.0
        return vx, vy, ax, ay, confidence, float(np.hypot(vx, vy))

    @classmethod
    def _entity_features(
        cls,
        candidates: list[tuple[float, Mapping[str, Any], Deque[tuple[float, float, float]]]],
        player: tuple[float, float],
        arena: tuple[float, float],
        speed_scale: float,
    ) -> tuple[np.ndarray, int]:
        output = np.zeros(TRAJECTORY_ENTITY_FEATURES, dtype=np.float32)
        if not candidates:
            return output, 0
        px, py = player
        width, height = arena
        ranked = []
        tracked_count = 0
        for _, entity, samples in candidates:
            ex, ey = _xy(entity.get("position"))
            vx, vy, ax, ay, confidence, speed = cls._motion(samples, entity)
            horizon = cls.horizon_seconds
            predicted_x = ex + vx * horizon + 0.5 * ax * horizon * horizon
            predicted_y = ey + vy * horizon + 0.5 * ay * horizon * horizon
            predicted_distance = float(np.hypot(predicted_x - px, predicted_y - py))
            relative_distance = max(1.0, float(np.hypot(ex - px, ey - py)))
            approaching = float(np.clip(
                (-(ex - px) * vx - (ey - py) * vy)
                / max(1.0, relative_distance * speed_scale),
                -1.0,
                1.0,
            ))
            if confidence >= 0.5:
                tracked_count += 1
            rank = predicted_distance - max(0.0, approaching) * speed_scale * 0.35
            ranked.append((rank, predicted_x, predicted_y, vx, vy, ax, ay, speed, approaching, confidence))
        _, predicted_x, predicted_y, vx, vy, ax, ay, speed, approaching, confidence = min(
            ranked,
            key=lambda row: row[0],
        )
        output[0] = np.clip((predicted_x - px) / max(1.0, width), -1.0, 1.0)
        output[1] = np.clip((predicted_y - py) / max(1.0, height), -1.0, 1.0)
        output[2] = np.clip(vx / speed_scale, -1.0, 1.0)
        output[3] = np.clip(vy / speed_scale, -1.0, 1.0)
        output[4] = np.clip(speed / speed_scale, 0.0, 1.0)
        output[5] = np.clip(float(np.hypot(ax, ay)) / cls.acceleration_scale, 0.0, 1.0)
        output[6] = np.clip(approaching, -1.0, 1.0)
        output[7] = confidence
        return output, tracked_count

    def features(self, state: Mapping[str, Any]) -> np.ndarray:
        session = str(state.get("session", ""))
        tick = int(_number(state.get("tick"), -1.0))
        identity = (session, tick)
        if self._last_identity == identity and tick >= 0:
            return self._last_features.copy()
        if self._last_identity is not None and session != self._last_identity[0]:
            self.reset()
        timestamp = self._timestamp(state, self._last_timestamp)
        self._last_identity = identity
        self._last_timestamp = timestamp
        player = _mapping(state.get("player"))
        arena = _mapping(state.get("arena"))
        player_position = _xy(player.get("position"))
        arena_size = (
            max(1.0, _number(arena.get("width"), 1920.0)),
            max(1.0, _number(arena.get("height"), 1080.0)),
        )
        active_keys: set[tuple[str, str]] = set()
        candidates: dict[str, list[tuple[float, Mapping[str, Any], Deque[tuple[float, float, float]]]]] = {
            "projectile": [],
            "enemy": [],
        }
        for kind, value in (("projectile", state.get("projectiles")), ("enemy", state.get("enemies"))):
            for entity in _items(value):
                key = self._entity_key(kind, entity)
                if not key:
                    continue
                position = _xy(entity.get("position"))
                track_key = (kind, key)
                active_keys.add(track_key)
                track = self._tracks.setdefault(
                    track_key,
                    {"samples": deque(maxlen=3), "last_seen": timestamp},
                )
                samples = track["samples"]
                if not samples or float(np.hypot(samples[-1][1] - position[0], samples[-1][2] - position[1])) > 0.01:
                    samples.append((timestamp, position[0], position[1]))
                track["last_seen"] = timestamp
                distance = float(np.hypot(position[0] - player_position[0], position[1] - player_position[1]))
                candidates[kind].append((distance, entity, samples))
        stale = [
            key for key, track in self._tracks.items()
            if key not in active_keys
            and timestamp - float(track.get("last_seen", timestamp)) > self.stale_after_seconds
        ]
        for key in stale:
            self._tracks.pop(key, None)
        projectile_features, projectile_tracked = self._entity_features(
            candidates["projectile"], player_position, arena_size, self.projectile_speed_scale
        )
        enemy_features, enemy_tracked = self._entity_features(
            candidates["enemy"], player_position, arena_size, self.enemy_speed_scale
        )
        output = np.zeros(TRAJECTORY_FEATURES, dtype=np.float32)
        output[TRAJECTORY_PROJECTILE_START:TRAJECTORY_ENEMY_START] = projectile_features
        output[TRAJECTORY_ENEMY_START:TRAJECTORY_PROJECTILE_COUNT] = enemy_features
        output[TRAJECTORY_PROJECTILE_COUNT] = np.clip(len(candidates["projectile"]) / 32.0, 0.0, 1.0)
        output[TRAJECTORY_PROJECTILE_TRACKED] = np.clip(projectile_tracked / 32.0, 0.0, 1.0)
        output[TRAJECTORY_ENEMY_COUNT] = np.clip(len(candidates["enemy"]) / 64.0, 0.0, 1.0)
        output[TRAJECTORY_ENEMY_TRACKED] = np.clip(enemy_tracked / 64.0, 0.0, 1.0)
        self._last_features = output
        return output.copy()


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
    # Discourage parking in the exact center when the arena is safe, while
    # leaving true hazard/recovery decisions unconstrained.
    center_stagnation_reward_scale = 0.012
    center_stagnation_radius = 0.12
    center_stagnation_threat_exemption = 0.45
    enemy_contact_guard = True
    enemy_contact_guard_threshold = 0.22
    enemy_contact_guard_margin = 0.08
    enemy_contact_override_penalty = 0.02
    # Small dense penalty; the unified hazard scorer remains the authority for
    # emergency movement and can override this when a safer lane exists.
    ranged_spacing_reward_scale = 0.015

    def __init__(self):
        self.base = BulletHellCombatVectorizer()
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_STEPS)
        self.trajectory_tracker = ThreeFrameTrajectoryTracker()
        self.previous_snapshot: dict[str, float] | None = None
        self.last_tick: tuple[str, int] | None = None
        self.reset()

    def reset(self, state: Mapping[str, Any] | None = None) -> None:
        self.history.clear()
        self.trajectory_tracker.reset()
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
        # These are risks for candidate actions, not nine simultaneous hits.
        # Using max() made one bad direction turn almost every state into an
        # evade objective, which explains the current ~99% evade telemetry.
        threat = _top_action_threat(combined)
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
            # Keep combat intent (engage) separate from movement intent.  A
            # ranged build should approach a stand-off point or orbit it, not
            # steer directly into the enemy center.
            combat = _mapping(state.get("combat"))
            ranged_count = _number(combat.get("ranged_count"), 0.0)
            melee_count = _number(combat.get("melee_count"), 0.0)
            weapon_range = _number(combat.get("weapon_range"), 0.0)
            if (
                target is not None
                and ranged_count > melee_count
                and weapon_range > 0.0
            ):
                tx, ty = _xy(target.get("position"))
                away_x, away_y = px - tx, py - ty
                distance = max(1.0, float(np.hypot(away_x, away_y)))
                stand_off = max(180.0, min(420.0, weapon_range * 0.55))
                if distance < stand_off * 0.92:
                    # At the band, use a tangent waypoint to keep motion
                    # flowing around the target instead of reversing into it.
                    tangent_x, tangent_y = -away_y / distance, away_x / distance
                    target = {
                        "position": {
                            "x": px + tangent_x * 180.0,
                            "y": py + tangent_y * 180.0,
                        }
                    }
                else:
                    target = {
                        "position": {
                            "x": tx + away_x / distance * stand_off,
                            "y": ty + away_y / distance * stand_off,
                        }
                    }
            urgency = 0.35
        else:
            objective = OBJECTIVE_REPOSITION
            # Pull back only when outside the safe band.  Inside it, leave
            # the actor free to orbit instead of making the exact center an
            # attractor.
            safe_x = min(max(px, width * 0.25), width * 0.75)
            safe_y = min(max(py, height * 0.25), height * 0.75)
            output[-3] = np.clip((safe_x - px) / width, -1.0, 1.0)
            output[-2] = np.clip((safe_y - py) / height, -1.0, 1.0)
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
        macro_start = cursor + HISTORY_SIZE
        output[macro_start:macro_start + MACRO_FEATURES] = self._macro(state)
        output[macro_start + MACRO_FEATURES:] = self.trajectory_tracker.features(state)
        return output
