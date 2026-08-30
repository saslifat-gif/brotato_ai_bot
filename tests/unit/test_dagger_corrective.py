"""Unit checks for the offline DAgger selector and state joins."""

import pytest

pytest.importorskip("numpy")

from v4.dagger_corrective import (
    _resolve_captured_state,
    _stable_holdout,
    priority_score,
)
from v4.dagger_review import render_review_html


def _record():
    return {
        "episode": 1,
        "tick": 42,
        "handcrafted_action": 3,
        "human_model_proposal": 7,
        "model_confidence": 0.93,
        "dangerous_state": True,
        "context": {
            "health_fraction": 0.14,
            "enemy_count": 12,
            "projectile_count": 8,
            "nearest_enemy_distance": 75.0,
        },
        "handcrafted_risk": {"total": 0.20},
        "human_risk": {"total": 0.91},
        "human_minus_handcrafted_risk": 0.71,
        "safety": {"human_would_override": True},
    }


def test_priority_score_explains_high_value_bot_state():
    score, reasons = priority_score(_record())
    assert score > 15.0
    assert "learned_vs_handcrafted_disagreement" in reasons
    assert "high_confidence" in reasons
    assert "counterfactual_safety_override" in reasons
    assert "learned_much_riskier" in reasons
    assert "very_low_hp" in reasons
    assert "dense_combat" in reasons


def test_sidecar_join_expands_and_orders_exact_history():
    record = {
        "state_ref": "current",
        "temporal_history": [
            {"state_ref": "newer", "timestamp_ms": 300.0},
            {"state_ref": "older", "timestamp_ms": 100.0},
        ],
    }
    lookup = {
        "current": {"state_ref": "current", "timestamp_ms": 400.0, "state": {"tick": 4}},
        "newer": {"state_ref": "newer", "timestamp_ms": 300.0, "state": {"tick": 3}},
        "older": {"state_ref": "older", "timestamp_ms": 100.0, "state": {"tick": 1}},
    }
    state, history = _resolve_captured_state(record, lookup)
    assert state == {"tick": 4}
    assert [sample["state"]["tick"] for sample in history] == [1, 3]
    assert [sample["timestamp_ms"] for sample in history] == [100.0, 300.0]


def test_holdout_assignment_is_stable_and_binary():
    first = _stable_holdout("same-queue-id", 0.25)
    assert first in {"train", "holdout"}
    assert first == _stable_holdout("same-queue-id", 0.25)


def test_visual_review_contains_captured_state_and_no_labels(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        '{"queue_id":"q1","split":"train","source_episode":"ep1",'
        '"source_tick":3,"source_timestamp_ms":1000,"state":{"arena":{"width":100,"height":80},'
        '"player":{"position":{"x":50,"y":40}},"enemies":[]},'
        '"selection_reasons":["high_confidence"],"current_action":1,'
        '"handcrafted_recommendation":1,"learned_proposal":2,"model_confidence":0.9}\n',
        encoding="utf-8",
    )
    output = tmp_path / "review.html"
    report = render_review_html(queue, output)
    html = output.read_text(encoding="utf-8")
    assert report["rows"] == 1
    assert report["labels_created"] == 0
    assert "q1" in html
    assert "Recorded battlefield tactical map" in html
    assert "human_corrective_action" in html
    assert "synthetic_labels" not in html
