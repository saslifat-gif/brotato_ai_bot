"""Parity between live human-policy features and the training representation.

The strongest available parity check: build a real demonstration dataset in a
temporary SQLite file through ``HumanDemoWriter`` (the production recorder
path), then compare ``v3_event_human_bc.build_examples`` inputs (training) with
``HumanPolicyFeatureBuilder`` inputs (live inference) for the same state
sequence.
"""

import sqlite3

import numpy as np
import pytest

pytest.importorskip("torch")

from brotato_ai.data.human_demo import HumanDemoWriter
from brotato_ai.policy.features import HumanPolicyFeatureBuilder


def _state(published_ms, tick, action, offset=0.0):
    return {
        "session": "parity-session",
        "tick": tick,
        "phase": "combat",
        "published_at_ms": published_ms,
        "human_action": action,
        "human_input": {"x": 0.0, "y": 0.0},
        "arena": {"width": 1920, "height": 1080},
        "wave": {"number": 3},
        "player": {
            "position": {"x": 900.0 + offset, "y": 540.0 - offset},
            "velocity": {"x": 10.0, "y": -5.0},
            "health": 12, "max_health": 15, "radius": 28,
        },
        "combat": {
            "weapon_count": 2, "melee_count": 1, "ranged_count": 1,
            "weapon_range": 220, "move_speed": 320, "armor": 4, "attack_speed": 15,
        },
        "enemies": [
            {
                "id": f"enemy-{index}", "type": "charger", "runtime_id": f"e{index}",
                "position": {"x": 700.0 + 60.0 * index + offset, "y": 500.0},
                "velocity": {"x": -30.0, "y": 0.0},
                "health": 5, "max_health": 6, "radius": 30,
                "attack_method": "contact",
            }
            for index in range(3)
        ],
        "projectiles": [
            {
                "id": "proj-0", "runtime_id": "p0", "hostile": True,
                "position": {"x": 1000.0 + offset, "y": 480.0},
                "velocity": {"x": -260.0, "y": 0.0}, "radius": 12,
            }
        ],
        "attack_indicators": [],
        "pickups": [],
    }


def _write_dataset(path):
    actions = [4, 4, 4, 8, 8, 8, 2, 2, 4, 4]
    with HumanDemoWriter(path, session_id="parity") as writer:
        for frame_number, action in enumerate(actions):
            state = _state(
                published_ms=1_000.0 + 42.0 * frame_number,
                tick=100 + frame_number,
                action=action,
                offset=3.0 * frame_number,
            )
            writer.record_frame(state)
    return actions


def test_live_builder_matches_training_example_inputs(tmp_path):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import v3_event_human_bc as event_bc

    dataset = tmp_path / "parity.sqlite"
    _write_dataset(dataset)

    frames = event_bc.load_frames(dataset)
    assert frames, "recorder produced no feature rows"
    examples = event_bc.build_examples(frames)

    builder = HumanPolicyFeatureBuilder()
    for frame, example in zip(frames, examples):
        builder.observe(
            frame.state,
            frame.previous_action,
            timestamp_ms=frame.timestamp_ns / 1e6,
        )
        live_input = builder.build_input(frame.previous_action)
        assert live_input.shape == example.input.shape
        np.testing.assert_allclose(
            live_input, example.input, rtol=0.0, atol=1e-5,
            err_msg=f"parity drift at frame {frame.frame_number}",
        )
    assert builder.input_width == examples[0].input.shape[0]


def test_builder_exposes_previous_action_slice_state(tmp_path):
    from brotato_ai.policy.features import EVENT_PREVIOUS_ACTION_SLICE, zero_previous_action_slice

    features = np.ones(832, dtype=np.float32)
    zeroed = zero_previous_action_slice(features)
    assert zeroed[EVENT_PREVIOUS_ACTION_SLICE].sum() == 0.0
    assert features[EVENT_PREVIOUS_ACTION_SLICE].sum() == 9.0
    # Other slices are untouched.
    assert zeroed[0] == 1.0 and zeroed[25] == 1.0


def test_builder_rejects_non_monotonic_time_drift(tmp_path):
    """Timestamps equal to the previous sample must not corrupt selection."""

    builder = HumanPolicyFeatureBuilder()
    state = _state(1000.0, 1, 0)
    first = builder.observe(state, 0, timestamp_ms=100.0)
    duplicate = builder.observe(state, 0, timestamp_ms=90.0)
    assert np.array_equal(first, duplicate)
    assert len(builder) == 2


def test_recorded_features_use_held_action_for_previous_slice(tmp_path):
    """The stored semantic vector must carry the held action in slice 16:25."""

    dataset = tmp_path / "held.sqlite"
    _write_dataset(dataset)
    connection = sqlite3.connect(str(dataset))
    rows = connection.execute(
        "SELECT action, previous_action, feature_blob FROM frames ORDER BY frame_number"
    ).fetchall()
    connection.close()
    assert rows
    from brotato_ai.data.human_demo import _from_blob

    for action, previous_action, blob in rows:
        features = np.asarray(_from_blob(blob, []), dtype=np.float32)
        assert features.size == 832
        if action != previous_action:
            # The one-hot inside the raw semantic vector reflects the held
            # action; the event model zeroes it and re-adds it explicitly.
            assert features[16 + previous_action] == pytest.approx(1.0)
