"""Checkpoint save/load and fail-safe inference for the event human policy."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brotato_ai.policy.human_action import (
    EVENT_CHECKPOINT_FORMAT,
    EVENT_POLICY_SCHEMA_VERSION,
    EventHumanActionPolicy,
    EventHumanModel,
    HumanPolicyError,
    load_event_checkpoint,
    save_event_checkpoint,
)

INPUT_WIDTH = 33  # 24-state + 9 actions for a tiny test model


def _tiny_checkpoint(tmp_path, *, schema_version=EVENT_POLICY_SCHEMA_VERSION):
    model = EventHumanModel(INPUT_WIDTH, 9)
    mean = np.zeros(24, dtype=np.float32)
    std = np.ones(24, dtype=np.float32)
    path = save_event_checkpoint(
        tmp_path / "event_model.pt",
        model=model,
        mean=mean,
        std=std,
        change_threshold=0.5,
        metrics={"teacher_forced": {"f1": 0.14}},
        dataset="session_test.sqlite",
        seed=7,
        action_names=("IDLE", "UP", "DOWN", "LEFT", "RIGHT",
                      "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"),
        history_offsets_ms=(0.0, 200.0, 400.0),
        previous_action_slice=(16, 25),
    )
    return path, model


def _model_input(held_action=4, width=INPUT_WIDTH):
    values = np.zeros(width, dtype=np.float32)
    values[24 + held_action] = 1.0
    return values


def test_checkpoint_round_trip_and_proposal(tmp_path):
    path, _ = _tiny_checkpoint(tmp_path)
    policy = EventHumanActionPolicy.load(path)
    assert policy.input_width == INPUT_WIDTH
    assert policy.action_count == 9
    assert policy.change_threshold == pytest.approx(0.5)
    proposal = policy.propose(_model_input(held_action=4), held_action=4)
    assert proposal is not None
    assert proposal.action != 4
    assert 0 <= proposal.action < 9
    assert proposal.probabilities.shape == (9,)
    assert proposal.probability == pytest.approx(float(proposal.probabilities[proposal.action]))
    assert 0.0 <= proposal.change_probability <= 1.0
    assert 0.0 <= proposal.duration_ms <= policy.max_hold_ms
    assert policy.proposal_count == 1 and policy.failure_count == 0
    assert proposal.metadata["feature_schema_version"] >= 1


def test_checkpoint_rejects_wrong_schema_version(tmp_path):
    path, _ = _tiny_checkpoint(tmp_path)
    payload = load_event_checkpoint(path)
    payload["checkpoint_schema_version"] = EVENT_POLICY_SCHEMA_VERSION + 1
    broken = tmp_path / "mismatch.pt"
    torch.save(payload, broken)
    with pytest.raises(HumanPolicyError, match="schema mismatch"):
        EventHumanActionPolicy.load(broken)


def test_checkpoint_rejects_unknown_format(tmp_path):
    broken = tmp_path / "foreign.pt"
    torch.save({"format": "something_else"}, broken)
    with pytest.raises(HumanPolicyError, match="not a"):
        load_event_checkpoint(broken)


def test_checkpoint_rejects_corrupt_file_and_missing_file(tmp_path):
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch file at all")
    with pytest.raises(HumanPolicyError):
        load_event_checkpoint(corrupt)
    with pytest.raises(HumanPolicyError, match="not found"):
        load_event_checkpoint(tmp_path / "absent.pt")


def test_propose_never_raises_on_bad_input(tmp_path):
    path, _ = _tiny_checkpoint(tmp_path)
    policy = EventHumanActionPolicy.load(path)
    assert policy.propose(np.zeros(3, dtype=np.float32), held_action=0) is None
    assert policy.failure_count == 1
    # A valid call still works afterwards: failure is per-call, not fatal.
    assert policy.propose(_model_input(held_action=2), held_action=2) is not None


def test_propose_swallows_nan_outputs(tmp_path):
    path, _ = _tiny_checkpoint(tmp_path)
    policy = EventHumanActionPolicy.load(path)
    original_forward = policy._model.forward
    policy._model.forward = lambda values: (
        torch.full((values.shape[0],), float("nan")),
        torch.full((values.shape[0], 9), float("nan")),
        torch.full((values.shape[0],), float("nan")),
    )
    try:
        assert policy.propose(_model_input(held_action=1), held_action=1) is None
    finally:
        policy._model.forward = original_forward
    assert policy.failure_count == 1


def test_checkpoint_payload_uses_documented_format(tmp_path):
    path, _ = _tiny_checkpoint(tmp_path)
    payload = load_event_checkpoint(path)
    assert payload["format"] == EVENT_CHECKPOINT_FORMAT
    assert payload["dataset"] == "session_test.sqlite"
    assert len(payload["normalization_mean"]) == 24
    assert payload["previous_action_slice"] == [16, 25]
