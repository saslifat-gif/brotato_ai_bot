"""Convert structured API state into a stable 256-value observation."""

import math
from typing import Any, Iterable, Mapping

import numpy as np


OBSERVATION_SIZE = 256
MAX_ENEMIES = 24
MAX_PROJECTILES = 16
MAX_PICKUPS = 14


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _vector(item: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = item.get(key, {})
    if not isinstance(value, Mapping):
        return 0.0, 0.0
    return _number(value.get("x")), _number(value.get("y"))


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


class ApiStateVectorizer:
    observation_size = OBSERVATION_SIZE

    def build(self, state: Mapping[str, Any], previous_action: int = 0) -> np.ndarray:
        output = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        arena = state.get("arena", {}) if isinstance(state.get("arena"), Mapping) else {}
        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        player = state.get("player", {}) if isinstance(state.get("player"), Mapping) else {}
        px, py = _vector(player, "position")
        pvx, pvy = _vector(player, "velocity")
        health = _number(player.get("health"))
        max_health = max(1.0, _number(player.get("max_health"), 1.0))
        wave = state.get("wave", {}) if isinstance(state.get("wave"), Mapping) else {}
        counters = state.get("counters", {}) if isinstance(state.get("counters"), Mapping) else {}
        enemies = _items(state.get("enemies"))
        projectiles = _items(state.get("projectiles"))
        pickups = _items(state.get("pickups"))

        output[:16] = np.asarray(
            [
                np.clip((px / width) * 2.0 - 1.0, -1.0, 1.0),
                np.clip((py / height) * 2.0 - 1.0, -1.0, 1.0),
                np.clip(pvx / 1000.0, -1.0, 1.0),
                np.clip(pvy / 1000.0, -1.0, 1.0),
                np.clip(health / max_health, 0.0, 1.0),
                np.clip(_number(wave.get("time_left")) / max(1.0, _number(wave.get("duration"), 60.0)), 0.0, 1.0),
                np.clip(_number(wave.get("number")) / 20.0, 0.0, 1.0),
                np.clip(_number(counters.get("materials")) / 500.0, 0.0, 1.0),
                np.clip(len(enemies) / 100.0, 0.0, 1.0),
                np.clip(len(projectiles) / 100.0, 0.0, 1.0),
                np.clip(len(pickups) / 50.0, 0.0, 1.0),
                1.0 if state.get("phase") == "combat" else 0.0,
                1.0 if state.get("dead") else 0.0,
                1.0 if state.get("victory") else 0.0,
                np.clip(int(previous_action) / 8.0, 0.0, 1.0),
                1.0,
            ],
            dtype=np.float32,
        )

        def nearest(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            return sorted(
                items,
                key=lambda item: (_vector(item, "position")[0] - px) ** 2
                + (_vector(item, "position")[1] - py) ** 2,
            )

        cursor = 16
        for enemy in nearest(enemies)[:MAX_ENEMIES]:
            ex, ey = _vector(enemy, "position")
            evx, evy = _vector(enemy, "velocity")
            enemy_health = _number(enemy.get("health"))
            enemy_max = max(1.0, _number(enemy.get("max_health"), 1.0))
            output[cursor : cursor + 5] = (
                np.clip((ex - px) / width, -1.0, 1.0),
                np.clip((ey - py) / height, -1.0, 1.0),
                np.clip(evx / 1000.0, -1.0, 1.0),
                np.clip(evy / 1000.0, -1.0, 1.0),
                np.clip(enemy_health / enemy_max, 0.0, 1.0),
            )
            cursor += 5

        cursor = 16 + MAX_ENEMIES * 5
        for projectile in nearest(projectiles)[:MAX_PROJECTILES]:
            qx, qy = _vector(projectile, "position")
            qvx, qvy = _vector(projectile, "velocity")
            if qvx == 0.0 and qvy == 0.0:
                rotation = _number(projectile.get("rotation"))
                qvx, qvy = math.cos(rotation), math.sin(rotation)
            output[cursor : cursor + 4] = (
                np.clip((qx - px) / width, -1.0, 1.0),
                np.clip((qy - py) / height, -1.0, 1.0),
                np.clip(qvx / 1000.0, -1.0, 1.0),
                np.clip(qvy / 1000.0, -1.0, 1.0),
            )
            cursor += 4

        cursor = 16 + MAX_ENEMIES * 5 + MAX_PROJECTILES * 4
        for pickup in nearest(pickups)[:MAX_PICKUPS]:
            ix, iy = _vector(pickup, "position")
            dx = np.clip((ix - px) / width, -1.0, 1.0)
            dy = np.clip((iy - py) / height, -1.0, 1.0)
            kind = str(pickup.get("kind", "item")).lower()
            kind_value = 1.0 if kind == "consumable" else 0.5
            distance = np.clip(math.hypot(float(dx), float(dy)), 0.0, 1.0)
            output[cursor : cursor + 4] = (dx, dy, kind_value, distance)
            cursor += 4
        return output
