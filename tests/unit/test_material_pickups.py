from copy import deepcopy

from brotato_ai.control.materials import material_progress, prefer_materials
from brotato_ai.control.arbiter import FinalActionArbiter
from brotato_ai.control.hazards import UnifiedHazardScorer
from brotato_ai.control.recovery import CrowdRecoveryGuard
from brotato_ai.domain.decisions import HazardRisk


def state():
    return {'phase': 'combat', 'arena': {'width': 1000, 'height': 1000},
            'player': {'position': {'x': 500, 'y': 500}, 'health': 100, 'max_health': 100},
            'pickups': [{'category': 'material', 'material_value': 1,
                         'position': {'x': 650, 'y': 500}}]}


def test_pipeline_moves_toward_money_and_logs_reason():
    shield = UnifiedHazardScorer()
    arbiter = FinalActionArbiter(safety_shield=shield,
                                crowd_recovery_guard=CrowdRecoveryGuard(shield=shield))
    current = state()
    before = deepcopy(current)
    trace = arbiter.apply(current, 3)
    assert trace.decision.applied_action == 4
    assert trace.source == 'material_pickup'
    assert current == before


def test_dangerous_money_does_not_override_safe_direction():
    risks = {a: HazardRisk(projectile=1.) for a in range(9)}
    risks[3] = HazardRisk()
    assert prefer_materials(state(), risks, 3) == 3


def test_low_health_and_non_money_preserve_original_action():
    risks = {a: HazardRisk() for a in range(9)}
    current = state()
    current['player']['health'] = 20
    assert prefer_materials(current, risks, 3) == 3
    current = state()
    current['pickups'][0]['category'] = 'healing'
    assert prefer_materials(current, risks, 3) == 3
    current['pickups'] = []
    assert prefer_materials(current, risks, 3) == 3


def test_far_money_is_ignored_and_already_approaching_is_stable():
    risks = {a: HazardRisk() for a in range(9)}
    assert prefer_materials(state(), risks, 4) == 4
    current = state()
    current['pickups'][0]['position']['x'] = 1000
    assert prefer_materials(current, risks, 3) == 3


def test_escape_prefers_money_only_between_safe_choices():
    guard = CrowdRecoveryGuard()
    payload = state()
    risks = {a: HazardRisk() for a in range(9)}
    guard._material_progress = material_progress(payload)
    assert guard._score_action(payload, risks, 4, side=1, previous_action=None) < guard._score_action(payload, risks, 3, side=1, previous_action=None)
    risks[4] = HazardRisk(projectile=.5)
    assert guard._score_action(payload, risks, 4, side=1, previous_action=None) > guard._score_action(payload, risks, 3, side=1, previous_action=None)

from brotato_ai.control.materials import MaterialTargetTracker


def test_persistent_target_does_not_cancel_opposing_coins():
    tracker = MaterialTargetTracker()
    payload = state()
    risks = {a: HazardRisk() for a in range(9)}
    assert tracker.apply(payload, risks, 0) == 4
    payload['pickups'].append({'category': 'material', 'position': {'x': 400, 'y': 500}})
    assert tracker.apply(payload, risks, 0) == 4
    payload['pickups'].pop(0)
    assert tracker.apply(payload, risks, 0) == 3
    tracker.reset()
    assert tracker.target is None


def test_blocked_target_is_abandoned_for_safe_coin():
    tracker = MaterialTargetTracker()
    payload = state()
    risks = {a: HazardRisk() for a in range(9)}
    assert tracker.apply(payload, risks, 0) == 4
    payload['pickups'].append({'category': 'material', 'position': {'x': 400, 'y': 500}})
    for a in (4, 6, 8):
        risks[a] = HazardRisk(projectile=1.)
    assert tracker.apply(payload, risks, 0) == 3


def test_recovery_collects_without_sacrificing_enemy_separation():
    tracker = MaterialTargetTracker()
    payload = state()
    risks = {a: HazardRisk() for a in range(9)}
    assert tracker.apply(payload, risks, 3, recovery=True) == 4
    payload['enemies'] = [{'position': {'x': 550, 'y': 500}, 'radius': 20}]
    assert tracker.apply(payload, risks, 3, recovery=True) == 3


def test_tracker_preserves_low_health_and_hazard_limits():
    tracker = MaterialTargetTracker()
    risks = {a: HazardRisk(projectile=1.) for a in range(9)}
    risks[3] = HazardRisk()
    assert tracker.apply(state(), risks, 3) == 3
    payload = state()
    payload['player']['health'] = 20
    assert tracker.apply(payload, {a: HazardRisk() for a in range(9)}, 3) == 3


def test_assertive_pickups_allow_spacing_but_not_collision_risk():
    payload = state()
    risks = {a: HazardRisk(projectile=1.) for a in range(9)}
    risks[0] = HazardRisk()
    risks[4] = HazardRisk(ranged_spacing=.25)
    assert MaterialTargetTracker().apply(payload, risks, 0) == 4
    for hazard in ('enemy', 'enemy_path', 'projectile', 'indicator', 'boundary'):
        risks[4] = HazardRisk(**{hazard: .25})
        assert MaterialTargetTracker().apply(payload, risks, 0) == 0


def test_assertive_tracker_can_seek_more_distant_coins():
    payload = state()
    payload['pickups'][0]['position']['x'] = 1100
    assert MaterialTargetTracker().apply(payload, {a: HazardRisk() for a in range(9)}, 0) == 4
