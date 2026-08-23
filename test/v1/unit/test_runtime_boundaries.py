import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "v1"))

from runtime.input_driver import InputDriver, normalize_input_mode
from runtime.state_machine import RuntimePhase, RuntimeStateMachine


class RuntimeBoundaryTests(unittest.TestCase):
    def test_legacy_input_names_are_normalized_without_fallbacks(self):
        self.assertEqual(normalize_input_mode("safe_background"), "background")
        self.assertEqual(normalize_input_mode("aggressive_click"), "physical_foreground")
        self.assertEqual(normalize_input_mode("nonsense"), "physical_foreground")

    def test_driver_keeps_one_selected_backend(self):
        self.assertEqual(InputDriver(1, "safe_background").input_mode, "background")
        self.assertEqual(InputDriver(1, "physical").input_mode, "physical_foreground")

    def test_phase_machine_prioritizes_menu_templates(self):
        machine = RuntimeStateMachine(non_battle_threshold=0.62)
        obs = machine.observe("unknown", 0.1, {"go": 0.91, "choose": 0.2, "restart": 0.1})
        self.assertEqual(obs.phase, RuntimePhase.SHOP)
        obs = machine.observe("restart", 0.8, {})
        self.assertEqual(obs.phase, RuntimePhase.GAMEOVER)

    def test_phase_machine_normalizes_battle_aliases(self):
        machine = RuntimeStateMachine()
        self.assertEqual(machine.observe("combat", 0.9).phase, RuntimePhase.BATTLE)
        self.assertEqual(machine.observe("level-up", 0.9).phase, RuntimePhase.UPGRADE)


if __name__ == "__main__":
    unittest.main()
