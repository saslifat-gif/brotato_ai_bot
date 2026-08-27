import pytest

from brotato_ai.data.schema import RecordSchemaError, normalize_raw_record, validate_schema_version
from brotato_ai.domain.state import StateSnapshot


def test_state_snapshot_normalizes_required_fields_and_is_immutable():
    snapshot = StateSnapshot.from_payload(
        {
            "tick": "7",
            "session": "demo",
            "phase": "combat",
            "published_at_ms": "1250",
            "arena": {"width": "1000", "height": 600},
            "player": {
                "position": {"x": 10, "y": 20},
                "velocity": {"x": 3, "y": 4},
                "radius": 21,
                "health": 9,
                "max_health": 10,
            },
            "wave": {"number": 4, "time_left": 12.5},
            "enemies": [
                {
                    "runtime_id": "boss-1",
                    "position": {"x": 40, "y": 50},
                    "velocity": {"x": -2, "y": 0},
                    "radius": 80,
                    "is_boss": True,
                    "attack_method": "charge",
                }
            ],
            "projectiles": [
                {"runtime_id": "p1", "owner_runtime_id": "boss-1", "hostile": True}
            ],
            "projectile_paths": {"enemy_action_risk": [0.3]},
        }
    )
    assert snapshot.tick == 7
    assert snapshot.player.velocity.x == 3
    assert snapshot.enemies[0].attack_method == "charge"
    assert snapshot.projectiles[0].owner_runtime_id == "boss-1"
    assert snapshot.path_risks.enemy[0] == pytest.approx(0.3)
    with pytest.raises(TypeError):
        snapshot.payload["tick"] = 8


def test_state_snapshot_documents_safe_missing_data_defaults():
    snapshot = StateSnapshot.from_payload({})
    assert snapshot.tick == -1
    assert snapshot.phase == "unknown"
    assert snapshot.arena_width == 1920
    assert snapshot.player.max_health == 1
    assert len(snapshot.path_risks.projectile) == 9
    assert snapshot.to_dict()["projectiles"] == []


def test_state_snapshot_accepts_legacy_scalar_wave_number():
    snapshot = StateSnapshot.from_payload({"wave": 17})
    assert snapshot.wave_number == 17
    assert snapshot.to_dict()["wave"]["number"] == 17


def test_raw_record_schema_is_explicit_and_rejects_unknown_versions():
    record = normalize_raw_record({"type": "raw_state", "tick": 1, "action": 3})
    assert record["schema_version"] == 2
    assert record["record_type"] == "state_snapshot"
    assert record["action"] == 3
    with pytest.raises(RecordSchemaError):
        validate_schema_version({"schema_version": 99})
