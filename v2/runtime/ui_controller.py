from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from v2.perception.types import Detection, FrameDetections


class UiAction(str, Enum):
    WAIT = "wait"
    RESTART = "restart"
    TAKE_ITEM = "take_item"
    PICK_UPGRADE = "pick_upgrade"
    NEXT_WAVE = "next_wave"


@dataclass(frozen=True)
class UiDecision:
    action: UiAction
    client_point: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    reason: str = ""


class UiController:
    """Turn detected UI buttons into safe clicks without hard-coded coordinates.

    Purchases are intentionally omitted until price/currency perception exists.
    The safe baseline can skip the shop and continue to the next wave.
    """

    _PRIORITY = (
        ("restart_button", UiAction.RESTART),
        ("take_item_button", UiAction.TAKE_ITEM),
        ("upgrade_card", UiAction.PICK_UPGRADE),
        ("next_wave_button", UiAction.NEXT_WAVE),
    )

    def __init__(self, minimum_confidence: float = 0.65):
        self.minimum_confidence = float(minimum_confidence)

    @staticmethod
    def _click_point(item: Detection) -> Tuple[int, int]:
        x, y = item.center
        return int(round(x)), int(round(y))

    def decide(self, detections: FrameDetections) -> UiDecision:
        for label, action in self._PRIORITY:
            item = detections.best(label)
            if item is None or float(item.confidence) < self.minimum_confidence:
                continue
            return UiDecision(
                action=action,
                client_point=self._click_point(item),
                confidence=float(item.confidence),
                reason=f"detected:{label}",
            )
        return UiDecision(UiAction.WAIT, reason="no_confirmed_ui_anchor")

