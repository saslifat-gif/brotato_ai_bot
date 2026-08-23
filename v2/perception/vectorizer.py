import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .types import Detection, FrameDetections


@dataclass(frozen=True)
class VectorLayout:
    enemy_slots: int = 12
    projectile_sectors: int = 8
    loot_slots: int = 4
    obstacle_sectors: int = 8
    action_count: int = 9

    @property
    def size(self) -> int:
        # player(x,y,visible), hp, wave + enemy(dx,dy,dist,conf) +
        # projectile(count,min_dist) + loot(dx,dy,dist) + obstacle min_dist + action one-hot
        return (
            5
            + self.enemy_slots * 4
            + self.projectile_sectors * 2
            + self.loot_slots * 3
            + self.obstacle_sectors
            + self.action_count
        )


class CombatStateVectorizer:
    """Convert detections into a fixed, resolution-independent policy input."""

    def __init__(self, layout: Optional[VectorLayout] = None):
        self.layout = layout or VectorLayout()
        self._last_player: Optional[Tuple[float, float]] = None

    @property
    def observation_size(self) -> int:
        return self.layout.size

    def reset(self) -> None:
        self._last_player = None

    @staticmethod
    def _nearest(items: Iterable[Detection], px: float, py: float):
        return sorted(items, key=lambda item: math.dist(item.center, (px, py)))

    @staticmethod
    def _relative(item: Detection, px: float, py: float, width: int, height: int):
        cx, cy = item.center
        dx = (cx - px) / max(1.0, float(width))
        dy = (cy - py) / max(1.0, float(height))
        dist = min(1.0, math.hypot(dx, dy) / math.sqrt(2.0))
        return float(dx), float(dy), float(dist)

    @staticmethod
    def _sector(dx: float, dy: float, count: int) -> int:
        angle = (math.atan2(dy, dx) + 2.0 * math.pi) % (2.0 * math.pi)
        return min(count - 1, int(angle / (2.0 * math.pi) * count))

    def build(
        self,
        detections: FrameDetections,
        hp_ratio: float = 1.0,
        wave_progress: float = 0.0,
        previous_action: int = 0,
    ) -> np.ndarray:
        width, height = detections.width, detections.height
        player = detections.best("player")
        player_visible = 1.0 if player is not None else 0.0
        if player is not None:
            px, py = player.center
            self._last_player = (px, py)
        elif self._last_player is not None:
            px, py = self._last_player
        else:
            px, py = width * 0.5, height * 0.5

        values = [
            float(np.clip(px / width, 0.0, 1.0)),
            float(np.clip(py / height, 0.0, 1.0)),
            player_visible,
            float(np.clip(hp_ratio, 0.0, 1.0)),
            float(np.clip(wave_progress, 0.0, 1.0)),
        ]

        enemies = self._nearest(detections.by_label("enemy"), px, py)
        for item in enemies[: self.layout.enemy_slots]:
            dx, dy, dist = self._relative(item, px, py, width, height)
            values.extend((dx, dy, dist, float(np.clip(item.confidence, 0.0, 1.0))))
        values.extend([0.0] * (self.layout.enemy_slots - min(len(enemies), self.layout.enemy_slots)) * 4)

        projectile_counts = np.zeros(self.layout.projectile_sectors, dtype=np.float32)
        projectile_distance = np.ones(self.layout.projectile_sectors, dtype=np.float32)
        for item in detections.by_label("projectile"):
            dx, dy, dist = self._relative(item, px, py, width, height)
            sector = self._sector(dx, dy, self.layout.projectile_sectors)
            projectile_counts[sector] += 1.0
            projectile_distance[sector] = min(projectile_distance[sector], dist)
        for count, distance in zip(projectile_counts, projectile_distance):
            values.extend((float(min(1.0, count / 10.0)), float(distance)))

        loot = self._nearest(detections.by_label("loot"), px, py)
        for item in loot[: self.layout.loot_slots]:
            values.extend(self._relative(item, px, py, width, height))
        values.extend([0.0] * (self.layout.loot_slots - min(len(loot), self.layout.loot_slots)) * 3)

        obstacle_distance = np.ones(self.layout.obstacle_sectors, dtype=np.float32)
        for item in detections.by_label("obstacle"):
            dx, dy, dist = self._relative(item, px, py, width, height)
            sector = self._sector(dx, dy, self.layout.obstacle_sectors)
            obstacle_distance[sector] = min(obstacle_distance[sector], dist)
        values.extend(float(v) for v in obstacle_distance)

        action_one_hot = np.zeros(self.layout.action_count, dtype=np.float32)
        if 0 <= int(previous_action) < self.layout.action_count:
            action_one_hot[int(previous_action)] = 1.0
        values.extend(float(v) for v in action_one_hot)

        out = np.asarray(values, dtype=np.float32)
        if out.shape != (self.observation_size,):
            raise RuntimeError(f"observation layout mismatch: got={out.shape} expected={self.observation_size}")
        return out

