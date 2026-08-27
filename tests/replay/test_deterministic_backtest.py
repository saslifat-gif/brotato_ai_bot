import json

from brotato_ai.data.replay import JsonlReplay
from brotato_ai.evaluation.backtest import compare_recording


def _record(tick, action, health=10, enemy_x=650):
    return {
        "type": "raw_state",
        "schema_version": 2,
        "tick": tick,
        "session": "fixture",
        "phase": "combat",
        "published_at_ms": tick * 50,
        "action": action,
        "arena": {"width": 1000, "height": 600},
        "player": {
            "position": {"x": 500, "y": 300},
            "velocity": {"x": 0, "y": 0},
            "health": health,
            "max_health": 10,
        },
        "wave": {"number": 4},
        "enemies": [
            {
                "runtime_id": "e1",
                "position": {"x": enemy_x, "y": 300},
                "velocity": {"x": -100, "y": 0},
                "radius": 60,
                "attack_method": "contact",
            }
        ],
        "projectiles": [],
        "projectile_paths": {
            "action_risk": [0] * 9,
            "enemy_action_risk": [0, 0, 0, 0, 0.9, 0, 0, 0, 0],
            "boundary_action_risk": [0] * 9,
        },
    }


def test_replay_is_ordered_and_deterministic(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [_record(1, 4), _record(2, 4, health=9), _record(3, 3, health=9)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = [(state.tick, action) for state, action in JsonlReplay(path).records()]
    second = [(state.tick, action) for state, action in JsonlReplay(path).records()]
    assert first == second == [(1, 4), (2, 4), (3, 3)]


def test_backtest_compares_required_variants_and_noop_detects_no_drift(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [_record(1, 4), _record(2, 4, health=9), _record(3, 4, health=9)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = compare_recording(path)
    second = compare_recording(path)
    assert first == second
    assert set(first["variants"]) == {
        "policy_only",
        "projectile_only",
        "enemy_only",
        "unified",
        "unified_stable",
        "noop_analyzer_control",
    }
    assert first["analyzer_drift"] == 0
    assert first["variants"]["unified"]["mean_modeled_risk"] < first["variants"]["policy_only"]["mean_modeled_risk"]
    assert first["observed_outcomes"]["damage_samples"] == 1
    assert first["observed_outcomes"]["unique_damage_windows"] == 1

