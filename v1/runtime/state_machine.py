"""Explicit phase state for the runtime.

Detection can flicker for a frame.  The state machine is intentionally small:
it normalizes detector/template names, applies confidence gates, and gives the
environment one phase value to use for both input and reward decisions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class RuntimePhase(str, Enum):
    UNKNOWN = "unknown"
    ALIGN = "align"
    PAUSED = "paused"
    BATTLE = "battle"
    SHOP = "shop"
    UPGRADE = "upgrade"
    ITEM_PICK = "item_pick"
    GAMEOVER = "gameover"


@dataclass(frozen=True)
class PhaseObservation:
    phase: RuntimePhase
    detector_state: str
    detector_score: float
    template_score: float


class RuntimeStateMachine:
    """Convert raw state signals into one stable runtime phase."""

    _ALIASES = {
        "combat": RuntimePhase.BATTLE,
        "fight": RuntimePhase.BATTLE,
        "wave": RuntimePhase.BATTLE,
        "go": RuntimePhase.SHOP,
        "store": RuntimePhase.SHOP,
        "merchant": RuntimePhase.SHOP,
        "choose": RuntimePhase.UPGRADE,
        "levelup": RuntimePhase.UPGRADE,
        "level_up": RuntimePhase.UPGRADE,
        "pick": RuntimePhase.ITEM_PICK,
        "reward": RuntimePhase.ITEM_PICK,
        "restart": RuntimePhase.GAMEOVER,
        "game_over": RuntimePhase.GAMEOVER,
        "death": RuntimePhase.GAMEOVER,
        "dead": RuntimePhase.GAMEOVER,
        "defeat": RuntimePhase.GAMEOVER,
    }

    def __init__(self, non_battle_threshold: float = 0.62, hysteresis_sec: float = 0.75):
        self.non_battle_threshold = float(non_battle_threshold)
        self.hysteresis_sec = max(0.0, float(hysteresis_sec))
        self.phase = RuntimePhase.UNKNOWN

    @classmethod
    def normalize(cls, name: str) -> RuntimePhase:
        raw = str(name or "").strip().lower().replace("-", "_")
        if raw in cls._ALIASES:
            return cls._ALIASES[raw]
        try:
            return RuntimePhase(raw)
        except ValueError:
            return RuntimePhase.UNKNOWN

    def reset(self) -> None:
        self.phase = RuntimePhase.UNKNOWN

    def observe(
        self,
        detector_state: str,
        detector_score: float,
        template_scores: Mapping[str, float] | None = None,
    ) -> PhaseObservation:
        scores = template_scores or {}
        template_score = max((float(v) for v in scores.values()), default=0.0)
        phase = self.normalize(detector_state)
        if phase == RuntimePhase.UNKNOWN and template_score >= self.non_battle_threshold:
            if float(scores.get("restart", 0.0)) >= self.non_battle_threshold:
                phase = RuntimePhase.GAMEOVER
            elif float(scores.get("choose", 0.0)) >= self.non_battle_threshold:
                phase = RuntimePhase.UPGRADE
            elif float(scores.get("go", 0.0)) >= self.non_battle_threshold:
                phase = RuntimePhase.SHOP
        self.phase = phase
        return PhaseObservation(
            phase=phase,
            detector_state=str(detector_state or "unknown"),
            detector_score=float(detector_score),
            template_score=template_score,
        )
