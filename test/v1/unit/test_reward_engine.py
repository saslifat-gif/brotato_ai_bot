import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "v1"))

from reward.reward_engine import RewardEngine


class RewardEngineTests(unittest.TestCase):
    def test_reward_components_are_finite(self):
        eng = RewardEngine(
            alive_reward=0.03,
            non_battle_penalty=0.2,
            damage_scale=8.0,
            idle_penalty=0.1,
            activity_bonus=0.05,
            loot_bonus=0.06,
            death_penalty=40.0,
            idle_diff_threshold=1.5,
            loot_delta_trigger=0.006,
        )
        out = eng.compute(
            prev_hp=1.0,
            curr_hp=0.9,
            is_battle=True,
            obs_diff=2.0,
            loot_delta=0.01,
            dead=False,
        )
        self.assertTrue(isinstance(out.total, float))
        self.assertIn("total", eng.reward_components_last)
        self.assertIn("total", eng.episode_component_sums)


if __name__ == "__main__":
    unittest.main()
