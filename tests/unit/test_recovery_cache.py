from collections import Counter

from brotato_ai.control import CombatSafetyShield, CrowdRecoveryGuard
from brotato_ai.control import recovery
from brotato_ai.domain.state import StateSnapshot


def scene():
    return {
        'phase': 'combat', 'tick': 1, 'wave': {'number': 10},
        'arena': {'width': 2048, 'height': 1536},
        'player': {'position': {'x': 1024, 'y': 768}, 'health': 20, 'max_health': 20},
        'enemies': [{'position': {'x': 1050+i*2, 'y': 768},
                     'velocity': {'x': -100, 'y': 0}, 'radius': 30} for i in range(24)],
        'combat': {'move_speed': 472},
    }


def test_recovery_geometry_is_cached_only_within_one_decision(monkeypatch):
    calls = Counter()
    original = recovery.enemy_separation_diagnostics
    def counted(payload, action):
        calls[action] += 1
        return original(payload, action)
    monkeypatch.setattr(recovery, 'enemy_separation_diagnostics', counted)
    guard = CrowdRecoveryGuard()
    payload = scene()
    guard.apply(payload, 4, control_interval_ms=42)
    assert calls and max(calls.values()) == 1
    assert guard._step_geometry is None
    calls.clear()
    # Same object and same tick: changed geometry must not reuse prior results.
    payload['player']['position']['x'] = 1100
    guard.apply(payload, 3, control_interval_ms=42)
    assert calls and max(calls.values()) == 1
    assert guard._step_geometry is None


def test_readonly_snapshot_scoring_matches_mapping_without_mutation():
    shield = CombatSafetyShield()
    snapshot = StateSnapshot.from_payload(scene())
    before = snapshot.to_dict()
    assert shield.all_risks(snapshot) == shield.all_risks(before)
    guard = CrowdRecoveryGuard(shield=shield)
    guard.apply(snapshot, 4, control_interval_ms=42)
    assert snapshot.to_dict() == before
