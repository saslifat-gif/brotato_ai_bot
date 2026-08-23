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
    anchor_confirmed: bool


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

    def __init__(
        self,
        shop_anchor_threshold: float = 0.72,
        upgrade_anchor_threshold: float = 0.70,
        gameover_anchor_threshold: float = 0.58,
        hysteresis_sec: float = 0.75,
    ):
        self.shop_anchor_threshold = float(shop_anchor_threshold)
        self.upgrade_anchor_threshold = float(upgrade_anchor_threshold)
        self.gameover_anchor_threshold = float(gameover_anchor_threshold)
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
        detected = self.normalize(detector_state)
        restart_hit = float(scores.get("restart", 0.0)) >= self.gameover_anchor_threshold
        choose_hit = float(scores.get("choose", 0.0)) >= self.upgrade_anchor_threshold
        go_hit = float(scores.get("go", 0.0)) >= self.shop_anchor_threshold

        # Menu actions require their visible button anchor. A classifier-only
        # menu prediction is unsafe because overlays and tooltips can resemble
        # upgrade/shop screens while the player is actually dead.
        anchor_confirmed = False
        if restart_hit:
            phase = RuntimePhase.GAMEOVER
            anchor_confirmed = True
        elif choose_hit:
            phase = RuntimePhase.ITEM_PICK if detected == RuntimePhase.ITEM_PICK else RuntimePhase.UPGRADE
            anchor_confirmed = True
        elif go_hit:
            phase = RuntimePhase.SHOP
            anchor_confirmed = True
        elif detected == RuntimePhase.BATTLE:
            phase = RuntimePhase.BATTLE
        else:
            phase = RuntimePhase.UNKNOWN
        self.phase = phase
        return PhaseObservation(
            phase=phase,
            detector_state=str(detector_state or "unknown"),
            detector_score=float(detector_score),
            template_score=template_score,
            anchor_confirmed=anchor_confirmed,
        )
