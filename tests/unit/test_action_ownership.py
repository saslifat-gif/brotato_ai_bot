from pathlib import Path

import pytest

from brotato_ai.bridge.client import BridgeClient
from brotato_ai.control import (
    CombatDecisionPipeline,
    CombatSafetyShield,
    CrowdRecoveryGuard,
    FinalActionWriter,
)


def _state(**updates):
    state = {
        "tick": 1,
        "session": "test",
        "phase": "combat",
        "published_at_ms": 1000,
        "arena": {"width": 1000, "height": 600},
        "player": {
            "position": {"x": 500, "y": 300},
            "velocity": {"x": 0, "y": 0},
            "health": 10,
            "max_health": 10,
        },
        "wave": {"number": 1},
        "enemies": [],
        "projectiles": [],
        "attack_indicators": [],
        "projectile_paths": {},
    }
    state.update(updates)
    return state


def _pipeline():
    shield = CombatSafetyShield()
    return CombatDecisionPipeline(
        safety_shield=shield,
        crowd_recovery_guard=CrowdRecoveryGuard(shield=shield),
    )


def test_normal_policy_action_produces_complete_trace():
    trace = _pipeline().apply(
        _state(), 4, previous_action=4, state_interval_ms=42, control_interval_ms=68
    )
    payload = trace.to_dict()
    assert payload["decision_source"] == "policy"
    assert payload["requested_action"] == payload["final_action"] == 4
    assert payload["state_interval_ms"] == 42
    assert payload["control_interval_ms"] == 68
    assert set(payload["requested_risk"]) >= {
        "total",
        "enemy",
        "projectile",
        "telegraph",
        "boundary",
    }


def test_conflicting_projectile_path_uses_one_hazard_override():
    paths = {
        "boundary_action_risk": [0] * 9,
        "enemy_action_risk": [0] * 9,
        "action_risk": [0, 0, 0, 0, 1.0, 0, 0, 0, 0],
    }
    trace = _pipeline().apply(
        _state(projectile_paths=paths), 4, previous_action=4
    )
    assert trace.source == "hazard"
    assert trace.hazard_overridden
    assert not trace.recovery_overridden
    assert trace.decision.applied_action != 4


def test_typed_snapshot_and_mapping_have_identical_risk_outputs():
    from brotato_ai.domain.state import StateSnapshot

    state = _state(
        enemies=[
            {
                "runtime_id": "enemy-1",
                "position": {"x": 610, "y": 300},
                "velocity": {"x": -40, "y": 0},
                "radius": 35,
                "attack_method": "charge",
            }
        ],
        projectiles=[
            {
                "runtime_id": "projectile-1",
                "position": {"x": 900, "y": 300},
                "velocity": {"x": -500, "y": 0},
                "radius": 12,
            }
        ],
    )
    scorer = CombatSafetyShield()
    mapping = scorer.all_risks(state)
    typed = scorer.all_risks(StateSnapshot.from_payload(state))
    reference = {
        action: scorer.risk_breakdown(state, action) for action in mapping
    }
    for action in mapping:
        assert typed[action].to_dict() == pytest.approx(mapping[action].to_dict())
        assert mapping[action].to_dict() == pytest.approx(reference[action].to_dict())


def test_vectorized_all_risks_matches_per_action_breakdown():
    enemies = [
        {
            "runtime_id": f"e{index}",
            "position": {"x": 500 + index * 12.0, "y": 280.0},
            "velocity": {"x": -40.0, "y": 15.0},
            "radius": 30 + index,
            "attack_method": "charge" if index % 3 == 0 else "projectile",
            "is_charging": index % 4 == 0,
            "is_boss": index == 0,
        }
        for index in range(30)
    ]
    projectiles = [
        {
            "runtime_id": f"p{index}",
            "position": {"x": 700 - index * 8.0, "y": 310.0},
            "velocity": {"x": -280.0, "y": 40.0 - index},
            "radius": 10,
            "hostile": True,
            "owner_runtime_id": "e0" if index < 4 else "e2",
        }
        for index in range(40)
    ]
    state = _state(
        wave={"number": 12},
        enemies=enemies,
        projectiles=projectiles,
        attack_indicators=[
            {
                "id": "aoe-1",
                "type": "warning",
                "position": {"x": 520, "y": 300},
                "width": 120,
                "height": 80,
                "time_to_activate": 0.4,
                "active": False,
                "owner_runtime_id": "e0",
            }
        ],
        projectile_paths={
            "action_risk": [0.1 * index for index in range(9)],
            "enemy_action_risk": [0.05 * index for index in range(9)],
            "boundary_action_risk": [0.02] * 9,
        },
    )
    scorer = CombatSafetyShield()
    mapping = scorer.all_risks(state)
    for action in mapping:
        assert mapping[action].to_dict() == pytest.approx(
            scorer.risk_breakdown(state, action).to_dict(), rel=1e-9, abs=1e-9
        )


def test_packed_enemies_are_not_scored_as_zero_risk():
    import math

    enemies = []
    for index in range(20):
        angle = index * 2.0 * math.pi / 20.0
        enemies.append(
            {
                "position": {
                    "x": 500.0 + 200.0 * math.cos(angle),
                    "y": 300.0 + 200.0 * math.sin(angle),
                },
                "velocity": {"x": 0.0, "y": 0.0},
                "radius": 30,
            }
        )
    scorer = CombatSafetyShield()
    idle = scorer.risk_breakdown(_state(wave={"number": 6}, enemies=enemies), 0)
    assert idle.enemy > 0.5
    assert scorer.all_risks(_state(wave={"number": 6}, enemies=enemies))[0].enemy == pytest.approx(
        idle.enemy
    )


def test_clearly_safer_lane_overrides_previous_action():
    from brotato_ai.domain.decisions import HazardRisk

    scorer = CombatSafetyShield()
    risks = {index: HazardRisk(enemy=0.50) for index in range(9)}
    risks[4] = HazardRisk(enemy=0.40)
    risks[1] = HazardRisk(enemy=0.10)
    decision = scorer.choose(risks, 4, previous_action=4)
    assert decision.applied_action == 1
    assert decision.overridden


def test_early_wave_pack_uses_policy_lane_not_escape():
    state = _state(
        wave={"number": 6},
        enemies=[{"position": {"x": 700, "y": 300}} for _ in range(18)],
    )
    trace = _pipeline().apply(state, 0, previous_action=0)
    assert trace.source != "crowd_recovery"
    assert trace.decision.applied_action != 0


def test_late_wave_pack_can_still_enter_escape():
    state = _state(
        wave={"number": 10},
        enemies=[{"position": {"x": 600, "y": 300}} for _ in range(18)],
    )
    trace = _pipeline().apply(state, 0, previous_action=0)
    assert trace.source == "crowd_recovery"
    assert trace.recovery_active


def test_emergency_recovery_is_explicit_and_reuses_scorer():
    state = _state(
        wave={"number": 19},
        enemies=[{"position": {"x": 600, "y": 300}} for _ in range(20)],
        projectile_paths={"boundary_action_risk": [0.9] * 9},
    )
    trace = _pipeline().apply(state, 0, previous_action=0)
    assert trace.source == "crowd_recovery"
    assert trace.recovery_active
    assert trace.decision.applied_action != 0


def test_pipeline_rejects_a_second_hazard_owner():
    first = CombatSafetyShield()
    second = CombatSafetyShield()
    with pytest.raises(ValueError, match="reuse"):
        CombatDecisionPipeline(
            safety_shield=first,
            crowd_recovery_guard=CrowdRecoveryGuard(shield=second),
        )


class FakeTransport:
    def __init__(self):
        self.messages = []

    def _write_final_action(self, action, sequence, timeout_sec=10.0):
        self.messages.append((action, sequence, timeout_sec))


def test_only_final_writer_emits_one_action():
    trace = _pipeline().apply(_state(), 2, previous_action=2)
    transport = FakeTransport()
    writer = FinalActionWriter(transport, timeout_sec=3)
    writer.write(trace, 12)
    assert transport.messages == [(2, 12, 3)]
    assert writer.write_count == 1


def test_bridge_rejects_direct_movement_send_without_connecting():
    with pytest.raises(RuntimeError, match="FinalActionWriter"):
        BridgeClient().send({"type": "action", "sequence": 1, "action": 4})


def test_runtime_environment_has_one_static_action_write_site():
    root = Path(__file__).resolve().parents[2]
    source = (root / "v3" / "env" / "brotato_api_env.py").read_text(encoding="utf-8")
    assert source.count("action_writer.write(") == 1
    assert "action_message(" not in source
    callback = (root / "v3" / "train_combat_finetune.py").read_text(encoding="utf-8")
    for tag in (
        "combat/hazard_applied_enemy_risk",
        "combat/hazard_applied_projectile_risk",
        "combat/hazard_source_crowd_recovery",
        "combat/hazard_control_interval_ms",
    ):
        assert tag in callback
