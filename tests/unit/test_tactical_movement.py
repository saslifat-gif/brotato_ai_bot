import numpy as np

from brotato_ai.control import CombatSafetyShield, CrowdRecoveryGuard
from brotato_ai.control.hazards import enemy_separation_diagnostics
from brotato_ai.domain.actions import MoveAction
from v4.combat_policy import HierarchicalCombatVectorizer


def state(*, enemy_x=650.0, enemy_y=300.0, enemy_vx=0.0, wave=1, paths=None, combat=None):
    return {
        "tick": 1,
        "session": "tactical-test",
        "phase": "combat",
        "arena": {"width": 1000.0, "height": 600.0},
        "player": {
            "position": {"x": 500.0, "y": 300.0},
            "velocity": {"x": 0.0, "y": 0.0},
            "health": 10.0,
            "max_health": 10.0,
            "radius": 28.0,
        },
        "wave": {"number": wave},
        "enemies": [
            {
                "runtime_id": "enemy-1",
                "position": {"x": enemy_x, "y": enemy_y},
                "velocity": {"x": enemy_vx, "y": 0.0},
                "radius": 40.0,
                "attack_method": "contact",
            }
        ],
        "projectiles": [],
        "attack_indicators": [],
        "projectile_paths": paths or {},
        "combat": combat or {},
    }


def test_enemy_motion_diagnostic_detects_approach_and_away_direction():
    approaching = state(enemy_vx=-100.0)
    away = enemy_separation_diagnostics(approaching, int(MoveAction.LEFT))
    toward = enemy_separation_diagnostics(approaching, int(MoveAction.RIGHT))
    assert away["active"]
    assert away["predicted_distance"] > away["current_distance"]
    assert away["radial_dot"] > toward["radial_dot"]


def test_escape_holds_through_one_safe_frame_and_then_releases():
    shield = CombatSafetyShield()
    controller = CrowdRecoveryGuard(shield=shield, hold_steps=3)
    threat = state(enemy_x=585.0, enemy_vx=-80.0)
    risks = shield.all_risks(threat)
    first = controller.apply(threat, int(MoveAction.RIGHT), risks=risks)
    assert controller.active
    assert first.applied_action != int(MoveAction.RIGHT)
    safe = state(enemy_x=900.0)
    safe_risks = shield.all_risks(safe)
    second = controller.apply(safe, int(MoveAction.LEFT), risks=safe_risks)
    assert controller.active
    controller.apply(safe, int(MoveAction.LEFT), risks=safe_risks)
    controller.apply(safe, int(MoveAction.LEFT), risks=safe_risks)
    released = controller.apply(safe, int(MoveAction.LEFT), risks=safe_risks)
    assert released.applied_action == int(MoveAction.LEFT)
    assert not controller.active


def test_boundary_emergency_keeps_legacy_trigger_without_center_attractor():
    shield = CombatSafetyShield()
    controller = CrowdRecoveryGuard(shield=shield)
    current = state(enemy_x=900.0, paths={"boundary_action_risk": [0.9] * 9})
    risks = shield.all_risks(current)
    decision = controller.apply(current, int(MoveAction.IDLE), risks=risks)
    assert controller.active
    assert decision.applied_action != int(MoveAction.IDLE)


def test_ranged_macro_targets_standoff_orbit_not_enemy_center():
    vectorizer = HierarchicalCombatVectorizer()
    current = state(
        enemy_x=560.0,
        combat={"ranged_count": 2, "melee_count": 0, "weapon_range": 500.0, "move_speed": 300.0},
    )
    macro = vectorizer._macro(current)
    assert np.isclose(macro[3], 1.0)
    # Macro movement target is encoded as a direction; it should point away
    # from the enemy once inside the ranged stand-off band.
    assert macro[-3] <= 0.0
    assert abs(float(macro[-2])) > 0.0


def test_tactical_state_is_not_permanently_escape():
    shield = CombatSafetyShield()
    controller = CrowdRecoveryGuard(shield=shield, hold_steps=2)
    threat = state(enemy_x=585.0)
    risks = shield.all_risks(threat)
    controller.apply(threat, int(MoveAction.RIGHT), risks=risks)
    clear = state(enemy_x=950.0)
    clear_risks = shield.all_risks(clear)
    controller.apply(clear, int(MoveAction.LEFT), risks=clear_risks)
    controller.apply(clear, int(MoveAction.LEFT), risks=clear_risks)
    controller.apply(clear, int(MoveAction.LEFT), risks=clear_risks)
    assert controller.state_name == controller.NORMAL
