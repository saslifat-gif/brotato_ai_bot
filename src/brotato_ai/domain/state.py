"""Normalized immutable state contract for live control and replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from collections.abc import Iterable, Mapping


STATE_SCHEMA_VERSION = 1


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _vector(value: Any) -> "Vector2":
    item = _mapping(value)
    return Vector2(_number(item.get("x")), _number(item.get("y")))


def _risk_vector(value: Any) -> tuple[float, ...]:
    output = [0.0] * 9
    if isinstance(value, (list, tuple)):
        for index, raw in enumerate(value[:9]):
            output[index] = min(1.0, max(0.0, _number(raw)))
    return tuple(output)


def _freeze(value: Any) -> Any:
    if type(value) in (int, float, str, bool, type(None)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Vector2:
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class PlayerSnapshot:
    position: Vector2
    velocity: Vector2
    radius: float
    health: float
    max_health: float


@dataclass(frozen=True)
class EnemySnapshot:
    runtime_id: str
    position: Vector2
    velocity: Vector2
    radius: float
    is_boss: bool
    attack_method: str


@dataclass(frozen=True)
class AttackSnapshot:
    runtime_id: str
    owner_runtime_id: str
    position: Vector2
    velocity: Vector2
    radius: float
    hostile: bool


@dataclass(frozen=True)
class PathRiskSnapshot:
    projectile: tuple[float, ...]
    enemy: tuple[float, ...]
    boundary: tuple[float, ...]


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable normalized view of one bridge state.

    Unknown bridge fields are retained in ``payload`` so observation adapters
    can evolve without changing the meaning of the typed fields.
    """

    schema_version: int
    timestamp_ms: int
    tick: int
    session: str
    phase: str
    arena_width: float
    arena_height: float
    player: PlayerSnapshot
    wave_number: int
    wave_time_left: float
    enemies: tuple[EnemySnapshot, ...]
    projectiles: tuple[AttackSnapshot, ...]
    telegraphs: tuple[AttackSnapshot, ...]
    path_risks: PathRiskSnapshot
    dead: bool
    victory: bool
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, value: Mapping[str, Any] | "StateSnapshot") -> "StateSnapshot":
        if isinstance(value, cls):
            return value
        raw = dict(_mapping(value))
        arena = _mapping(raw.get("arena"))
        player = _mapping(raw.get("player"))
        wave_value = raw.get("wave")
        wave = _mapping(wave_value)
        if not wave and wave_value is not None:
            wave = {"number": wave_value}
        paths = _mapping(raw.get("projectile_paths"))

        width = max(1.0, _number(arena.get("width"), 1920.0))
        height = max(1.0, _number(arena.get("height"), 1080.0))
        max_health = max(1.0, _number(player.get("max_health"), 1.0))
        position = _mapping(player.get("position"))
        normalized_player = PlayerSnapshot(
            position=Vector2(
                _number(position.get("x"), width * 0.5),
                _number(position.get("y"), height * 0.5),
            ),
            velocity=_vector(player.get("velocity")),
            radius=max(0.0, _number(player.get("radius"), 28.0)),
            health=max(0.0, _number(player.get("health"))),
            max_health=max_health,
        )
        enemies = tuple(
            EnemySnapshot(
                runtime_id=str(item.get("runtime_id", "")),
                position=_vector(item.get("position")),
                velocity=_vector(item.get("velocity")),
                radius=max(0.0, _number(item.get("radius"), 40.0)),
                is_boss=bool(item.get("is_boss")),
                attack_method=str(item.get("attack_method", "unknown")).strip().lower()
                or "unknown",
            )
            for item in _items(raw.get("enemies"))
        )

        def attacks(key: str) -> tuple[AttackSnapshot, ...]:
            return tuple(
                AttackSnapshot(
                    runtime_id=str(item.get("runtime_id", "")),
                    owner_runtime_id=str(item.get("owner_runtime_id", "")),
                    position=_vector(item.get("position")),
                    velocity=_vector(item.get("velocity")),
                    radius=max(0.0, _number(item.get("radius"), 12.0)),
                    hostile=bool(item.get("hostile", True)),
                )
                for item in _items(raw.get(key))
            )

        raw.setdefault("type", "state")
        raw.setdefault("protocol", 1)
        raw["schema_version"] = _integer(
            raw.get("schema_version"), STATE_SCHEMA_VERSION
        )
        raw["tick"] = _integer(raw.get("tick"), -1)
        raw["session"] = str(raw.get("session", ""))
        raw["phase"] = str(raw.get("phase", "unknown"))
        raw["published_at_ms"] = _integer(
            raw.get("published_at_ms", raw.get("timestamp_ms", -1)), -1
        )
        raw["arena"] = {**dict(arena), "width": width, "height": height}
        raw["player"] = {
            **dict(player),
            "position": {"x": normalized_player.position.x, "y": normalized_player.position.y},
            "velocity": {"x": normalized_player.velocity.x, "y": normalized_player.velocity.y},
            "radius": normalized_player.radius,
            "health": normalized_player.health,
            "max_health": normalized_player.max_health,
        }
        raw["wave"] = {
            **dict(wave),
            "number": _integer(wave.get("number")),
            "time_left": _number(wave.get("time_left")),
        }
        for key in ("enemies", "projectiles", "attack_indicators", "pickups"):
            raw[key] = [dict(item) for item in _items(raw.get(key))]
        raw.setdefault("combat", {})
        raw.setdefault("counters", {})
        raw.setdefault("ui", {})
        raw.setdefault("projectile_paths", {})
        raw["dead"] = bool(raw.get("dead"))
        raw["victory"] = bool(raw.get("victory"))

        return cls(
            schema_version=raw["schema_version"],
            timestamp_ms=raw["published_at_ms"],
            tick=raw["tick"],
            session=raw["session"],
            phase=raw["phase"],
            arena_width=width,
            arena_height=height,
            player=normalized_player,
            wave_number=_integer(wave.get("number")),
            wave_time_left=_number(wave.get("time_left")),
            enemies=enemies,
            projectiles=attacks("projectiles"),
            telegraphs=attacks("attack_indicators"),
            path_risks=PathRiskSnapshot(
                projectile=_risk_vector(paths.get("action_risk")),
                enemy=_risk_vector(paths.get("enemy_action_risk")),
                boundary=_risk_vector(paths.get("boundary_action_risk")),
            ),
            dead=raw["dead"],
            victory=raw["victory"],
            payload=_freeze(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)


def normalize_state(value: Mapping[str, Any] | StateSnapshot) -> dict[str, Any]:
    return StateSnapshot.from_payload(value).to_dict()
