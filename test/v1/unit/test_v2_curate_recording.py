import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v2.curate_recording import frame_change_score, next_distinct_index
from v2.import_obs_video import source_frame_indices
from v2.record_gameplay import visual_change_score


def _write_frame(path: Path, value: int) -> None:
    image = np.full((36, 64, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_frame_change_score_is_zero_for_identical_images():
    image = np.full((36, 64), 80, dtype=np.uint8)
    assert frame_change_score(image, image.copy()) == 0.0


def test_next_distinct_index_skips_static_samples(tmp_path: Path):
    frames = [tmp_path / f"frame_{index:08d}.jpg" for index in range(3)]
    _write_frame(frames[0], 40)
    _write_frame(frames[1], 40)
    _write_frame(frames[2], 180)

    index, skipped = next_distinct_index(frames, index=0, stride=1, min_change=2.0)

    assert index == 2
    assert skipped == 1


def test_visual_change_score_detects_motion():
    first = np.zeros((36, 64, 3), dtype=np.uint8)
    second = np.full((36, 64, 3), 255, dtype=np.uint8)

    assert visual_change_score(first, first.copy()) == 0.0
    assert visual_change_score(first, second) > 200.0


def test_obs_source_frame_indices_resample_to_target_fps():
    indices = list(source_frame_indices(source_fps=30.0, target_fps=5.0, total_frames=31))

    assert indices == [0, 6, 12, 18, 24, 30]
