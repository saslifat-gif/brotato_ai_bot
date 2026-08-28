import json
from pathlib import Path

from brotato_ai.evaluation.classify_live_failures import classify_recording


def _frame(tick: int, health: float, dead: bool = False, action: int = 4) -> dict:
    return {
        "type": "raw_state",
        "schema_version": 1,
        "phase": "combat",
        "tick": tick,
        "published_at_ms": tick * (1000.0 / 60.0),
        "session": "test",
        "action": action,
        "dead": dead,
        "wave": {"number": 3},
        "player": {"position": {"x": 500, "y": 300}, "health": health, "max_health": 10},
        "combat": {"move_speed": 300},
        "enemies": [{"position": {"x": 520, "y": 300}, "velocity": {"x": 0, "y": 0}, "radius": 40}],
        "projectiles": [
            {
                "position": {"x": 540, "y": 300},
                "velocity": {"x": -400, "y": 0},
                "radius": 12,
                "hostile": True,
            }
        ],
        "attack_indicators": [],
        "arena": {"width": 1000, "height": 600},
    }


def test_classify_recording_labels_a_death(tmp_path: Path) -> None:
    path = tmp_path / "live.jsonl"
    frames = [_frame(tick, 10.0) for tick in range(1, 20)]
    frames.append(_frame(20, 0.0, dead=True, action=4))
    path.write_text("\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")
    report = classify_recording(path)
    assert report["deaths_classified"] == 1
    assert report["events"][0]["category"] in {
        "too_late",
        "no_safe_action",
        "wrong_action",
        "already_best",
    }
