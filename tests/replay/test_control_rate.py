from brotato_ai.domain.state import StateSnapshot
from brotato_ai.evaluation.control_rate import run_rate


def _state(timestamp_ms: int, action: int = 3):
    return StateSnapshot.from_payload({
        "type": "raw_state", "schema_version": 2, "session": "s",
        "tick": timestamp_ms, "timestamp_ms": timestamp_ms, "phase": "combat",
        "arena": {"width": 1000, "height": 600},
        "player": {"position": {"x": 500, "y": 300}, "health": 10, "max_health": 10, "radius": 28},
        "wave": {"number": 1}, "enemies": [], "projectiles": [],
        "attack_indicators": [], "projectile_paths": {"action_risk": [0] * 9,
        "enemy_action_risk": [0] * 9, "boundary_action_risk": [0] * 9},
        "combat": {"move_speed": 300, "weapon_range": 170}, "action": action,
    })


def test_lower_rate_uses_latest_observation_and_holds_action():
    rows = [(_state(t, action=(t // 17) % 9), (t // 17) % 9) for t in (0, 17, 34, 51, 68, 85, 102)]
    low = run_rate(rows, 15.0)
    high = run_rate(rows, 60.0)
    assert low.decisions == 2
    assert high.decisions == 7
    assert low.actionable_delays_ms == []
    assert max(low.stale_observation_ms) > 0.0


def test_phase_offset_changes_only_schedule_not_recorded_frame_count():
    rows = [(_state(t), 3) for t in range(0, 101, 17)]
    first = run_rate(rows, 15.0, 0.0)
    shifted = run_rate(rows, 15.0, 33.333)
    assert first.frames == shifted.frames == len(rows)
    assert first.rate_hz == shifted.rate_hz == 15.0
