"""Escape holds must be real-time durations, not control-step counts.

The defaults reproduce the previous 8-step hold exactly at the 24 Hz bridge
rate; at any other rate the wall-clock hold stays stable instead of silently
scaling with the bridge rate (spec section 13).
"""

import pytest

from brotato_ai.control import CombatDecisionPipeline, CombatSafetyShield, CrowdRecoveryGuard

STEP_24HZ_MS = 1000.0 / 24.0


def state(*, enemy_x=650.0, enemy_y=300.0, enemy_vx=0.0, wave=1):
    return {
        "tick": 1,
        "session": "escape-timing-test",
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
        "projectile_paths": {},
        "combat": {},
    }


def _escape_then_safe_steps(controller, *, interval_ms, max_steps=60):
    """Enter escape, then apply safe states; return (held steps, last decision)."""

    shield = controller.shield
    threat = state(enemy_x=585.0, enemy_vx=-80.0)
    controller.apply(threat, 4, risks=shield.all_risks(threat), control_interval_ms=interval_ms)
    assert controller.active
    safe = state(enemy_x=900.0)
    safe_risks = shield.all_risks(safe)
    steps = 1
    for _ in range(max_steps):
        decision = controller.apply(safe, 1, risks=safe_risks, control_interval_ms=interval_ms)
        steps += 1
        if not controller.active:
            return steps, decision
    raise AssertionError("escape never released")


def test_default_hold_matches_step_behavior_at_24hz():
    step_based = CrowdRecoveryGuard(shield=CombatSafetyShield())
    duration_based = CrowdRecoveryGuard(shield=CombatSafetyShield())
    steps_step_based, _ = _escape_then_safe_steps(step_based, interval_ms=0.0)
    steps_duration_based, _ = _escape_then_safe_steps(duration_based, interval_ms=STEP_24HZ_MS)
    assert steps_step_based == steps_duration_based


def test_hold_releases_after_the_same_wall_clock_time_at_any_rate():
    steps_at_12hz, _ = _escape_then_safe_steps(
        CrowdRecoveryGuard(shield=CombatSafetyShield()), interval_ms=2 * STEP_24HZ_MS
    )
    steps_at_48hz, _ = _escape_then_safe_steps(
        CrowdRecoveryGuard(shield=CombatSafetyShield()), interval_ms=STEP_24HZ_MS / 2
    )
    wall_at_12hz = steps_at_12hz * 2 * STEP_24HZ_MS
    wall_at_48hz = steps_at_48hz * (STEP_24HZ_MS / 2)
    assert abs(wall_at_12hz - wall_at_48hz) <= 2 * STEP_24HZ_MS


def test_explicit_duration_overrides_step_derivation():
    controller = CrowdRecoveryGuard(
        shield=CombatSafetyShield(), hold_duration_ms=500.0
    )
    assert controller.hold_duration_ms == 500.0
    shield = controller.shield
    threat = state(enemy_x=585.0, enemy_vx=-80.0)
    controller.apply(threat, 4, risks=shield.all_risks(threat), control_interval_ms=STEP_24HZ_MS)
    assert controller.remaining_ms == pytest.approx(500.0 - STEP_24HZ_MS)


def test_remaining_ms_is_reported_by_the_arbiter():
    shield = CombatSafetyShield()
    arbiter = CombatDecisionPipeline(
        safety_shield=shield,
        crowd_recovery_guard=CrowdRecoveryGuard(shield=shield),
    )
    # The boundary emergency trigger enters escape regardless of the
    # hazard-chosen action the arbiter feeds into the guard.
    threat = state(enemy_x=900.0)
    threat["projectile_paths"] = {"boundary_action_risk": [0.9] * 9}
    trace = arbiter.apply(threat, 0, previous_action=1, control_interval_ms=STEP_24HZ_MS)
    assert trace.recovery_active
    assert trace.escape_remaining_ms == pytest.approx(333.333 - STEP_24HZ_MS, abs=1.0)
    assert trace.to_dict()["escape_remaining_ms"] == trace.escape_remaining_ms


def test_step_fallback_still_works_without_intervals():
    controller = CrowdRecoveryGuard(shield=CombatSafetyShield(), hold_steps=2)
    shield = controller.shield
    threat = state(enemy_x=585.0, enemy_vx=-80.0)
    controller.apply(threat, 4, risks=shield.all_risks(threat))
    assert controller.active
    safe = state(enemy_x=900.0)
    safe_risks = shield.all_risks(safe)
    controller.apply(safe, 1, risks=safe_risks)
    assert controller.active
    controller.apply(safe, 1, risks=safe_risks)
    assert not controller.active
