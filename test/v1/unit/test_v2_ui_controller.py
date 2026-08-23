import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v2.perception.types import Detection, FrameDetections
from v2.runtime.ui_controller import UiAction, UiController


class UiControllerTests(unittest.TestCase):
    def test_restart_has_priority_over_other_buttons(self):
        detections = FrameDetections.from_iterable(
            1920,
            1080,
            [
                Detection("next_wave_button", 0.99, (1500, 820, 1880, 900)),
                Detection("restart_button", 0.80, (500, 900, 800, 1000)),
            ],
        )
        decision = UiController().decide(detections)
        self.assertEqual(decision.action, UiAction.RESTART)
        self.assertEqual(decision.client_point, (650, 950))

    def test_low_confidence_button_is_not_clickable(self):
        detections = FrameDetections.from_iterable(
            1280,
            720,
            [Detection("upgrade_card", 0.40, (100, 100, 300, 400))],
        )
        decision = UiController(minimum_confidence=0.65).decide(detections)
        self.assertEqual(decision.action, UiAction.WAIT)
        self.assertIsNone(decision.client_point)


if __name__ == "__main__":
    unittest.main()
