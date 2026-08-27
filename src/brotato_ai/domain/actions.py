"""Movement actions and their normalized arena vectors."""

from __future__ import annotations

import math
from enum import IntEnum


class MoveAction(IntEnum):
    IDLE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    UP_LEFT = 5
    UP_RIGHT = 6
    DOWN_LEFT = 7
    DOWN_RIGHT = 8


ACTION_VECTORS: dict[MoveAction, tuple[float, float]] = {
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


def normalize_action(value: int | MoveAction) -> MoveAction:
    """Validate an external action before it enters the control pipeline."""

    return MoveAction(int(value))

