import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "health"))

from runtime.stop_manager import StopManager


class StopManagerTests(unittest.TestCase):
    def test_request_stop_sets_reason_once(self):
        sm = StopManager()
        self.assertFalse(sm.should_stop())
        self.assertTrue(sm.request_stop("SIGINT"))
        self.assertTrue(sm.should_stop())
        self.assertEqual(sm.reason(), "SIGINT")
        self.assertFalse(sm.request_stop("OTHER"))
        self.assertEqual(sm.reason(), "SIGINT")


if __name__ == "__main__":
    unittest.main()
