import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v2.perception.types import Detection, FrameDetections
from v2.perception.vectorizer import CombatStateVectorizer


class VectorizerTests(unittest.TestCase):
    def test_fixed_shape_and_finite_values(self):
        detections = FrameDetections.from_iterable(
            1920,
            1080,
            [
                Detection("player", 0.99, (940, 520, 980, 560)),
                Detection("enemy", 0.90, (1100, 520, 1140, 560)),
                Detection("projectile", 0.80, (1000, 540, 1010, 550)),
                Detection("loot", 0.70, (700, 700, 720, 720)),
            ],
        )
        vectorizer = CombatStateVectorizer()
        obs = vectorizer.build(detections, hp_ratio=0.75, wave_progress=0.5, previous_action=6)
        self.assertEqual(obs.shape, (98,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.isfinite(obs).all())
        self.assertAlmostEqual(float(obs[2]), 1.0)
        self.assertAlmostEqual(float(obs[3]), 0.75)

    def test_missing_player_uses_last_known_position(self):
        vectorizer = CombatStateVectorizer()
        first = FrameDetections.from_iterable(100, 100, [Detection("player", 1.0, (10, 20, 20, 30))])
        vectorizer.build(first)
        missing = FrameDetections.from_iterable(100, 100, [])
        obs = vectorizer.build(missing)
        self.assertAlmostEqual(float(obs[0]), 0.15)
        self.assertAlmostEqual(float(obs[1]), 0.25)
        self.assertAlmostEqual(float(obs[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
