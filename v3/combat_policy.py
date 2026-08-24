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

from v3.protocol import MoveAction


RICH_OBSERVATION_SIZE = 384
RICH_MAX_ENEMIES = 20
RICH_MAX_PROJECTILES = 20
RICH_MAX_PICKUPS = 8
SEMANTIC_OBSERVATION_SIZE = 832
SEMANTIC_MAX_INDICATORS = 10
SEMANTIC_MAX_WEAPONS = 6

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


class CombatSafetyShield:
    """Override only actions with an imminent geometric collision."""

    def __init__(self, *, enabled: bool = True, override_margin: float = 0.25):
        self.enabled = bool(enabled)
        self.override_margin = float(override_margin)

    def risk(self, state: Mapping[str, Any], action: int) -> float:
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
        risk = 0.0

        edge_margin = 150.0
        edge_distance = min(future_x, width - future_x, future_y, height - future_y)
        if edge_distance < edge_margin:
            risk += ((edge_margin - edge_distance) / edge_margin) ** 2 * 4.0

        for enemy in _nearest(_items(state.get("enemies")), px, py)[:24]:
            ex, ey = _xy(enemy.get("position"))
            evx, evy = _xy(enemy.get("velocity"))
            predicted_x = ex + evx * 0.45
            predicted_y = ey + evy * 0.45
            distance = math.hypot(predicted_x - future_x, predicted_y - future_y)
            radius = max(25.0, _number(enemy.get("radius"), 40.0))
            danger = radius + (210.0 if enemy.get("is_charging") else 100.0)
            if enemy.get("is_boss"):
                danger += 60.0
            if distance < danger:
                risk += ((danger - distance) / danger) ** 2 * (
                    5.0 if enemy.get("is_charging") else 2.0
                )

        for projectile in _nearest(_items(state.get("projectiles")), px, py)[:32]:
            tti, miss_distance = projectile_time_to_impact(
                projectile, (px, py), movement, player_speed
            )
            radius = max(8.0, _number(projectile.get("radius"), 12.0)) + 42.0
            qvx, qvy = _xy(projectile.get("velocity"))
            stationary = qvx * qvx + qvy * qvy < 100.0
            danger = radius * (2.3 if stationary else 1.5)
            if miss_distance < danger:
                urgency = 1.0 if stationary else 1.0 - min(1.0, tti / 0.8)
                risk += ((danger - miss_distance) / danger) ** 2 * (3.0 + 5.0 * urgency)
        return float(risk)

    def apply(self, state: Mapping[str, Any], requested_action: int) -> SafetyDecision:
        requested = int(MoveAction(int(requested_action)))
        if not self.enabled:
            return SafetyDecision(requested, requested, 0.0, 0.0)
        requested_risk = self.risk(state, requested)
        if requested_risk < 0.35:
            return SafetyDecision(requested, requested, requested_risk, requested_risk)
        scored = [(self.risk(state, int(action)), int(action)) for action in MoveAction]
        best_risk, best_action = min(scored, key=lambda row: (row[0], row[1] == 0))
        if requested_risk - best_risk < self.override_margin:
            best_action, best_risk = requested, requested_risk
        return SafetyDecision(requested, best_action, requested_risk, best_risk)


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
            output[cursor:cursor + 8] = (
                np.clip(dx / width, -1.0, 1.0), np.clip(dy / height, -1.0, 1.0),
                np.clip(vx / 1000.0, -1.0, 1.0), np.clip(vy / 1000.0, -1.0, 1.0),
                np.clip(radius / 300.0, 0.0, 1.0), np.clip(tti / 0.8, 0.0, 1.0),
                np.clip(miss / 500.0, 0.0, 1.0), 1.0 if vx * vx + vy * vy < 100.0 else 0.0,
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
    """V2 observation preserving v1 while adding API-only entity semantics."""

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
            attack_hash = _hash_pair(enemy.get("attack_type"))[0]
            movement_hash = _hash_pair(enemy.get("movement_type"))[0]
            output[cursor:cursor + 10] = (
                identity[0], identity[1],
                np.clip(_number(enemy.get("width"), 80.0) / width, 0.0, 1.0),
                np.clip(_number(enemy.get("height"), 80.0) / height, 0.0, 1.0),
                np.clip(_number(enemy.get("contact_damage")) / max_health, 0.0, 4.0) / 4.0,
                np.clip(_number(enemy.get("attack_cooldown_remaining")) / 5.0, 0.0, 1.0),
                1.0 if enemy.get("is_attacking") else 0.0,
                1.0 if enemy.get("is_elite") else 0.0,
                attack_hash,
                movement_hash,
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
        indicators = _nearest(_items(state.get("attack_indicators")), px, py)
        for indicator in indicators[:SEMANTIC_MAX_INDICATORS]:
            token_hash = _hash_pair(indicator.get("id"), indicator.get("type"))[0]
            dx, dy = _relative(indicator, px, py)
            direction_x, direction_y = _xy(indicator.get("direction"))
            output[cursor:cursor + 10] = (
                token_hash,
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


class SemanticCombatPolicyBase(nn.Module):
    """Small v2 actor that preserves v1 behavior and learns semantic residuals."""

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
    """Record API-semantic v2 observations with human movement labels."""

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
