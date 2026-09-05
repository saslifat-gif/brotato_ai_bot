from brotato_ai.control.safe_zone import SafeZonePlanner
from brotato_ai.domain.decisions import HazardRisk


def state():
    return {'phase': 'combat', 'arena': {'width': 1000, 'height': 1000},
            'player': {'position': {'x': 500, 'y': 500}},
            'combat': {'move_speed': 300}, 'enemies': [], 'projectiles': []}


def test_destination_persists_and_arrival_releases_it():
    planner = SafeZonePlanner()
    payload = state()
    risks = {a: HazardRisk() for a in range(9)}
    planner.apply(payload, risks, 4, 40)
    target = planner.target
    assert target is not None
    planner.apply(payload, risks, 3, 40)
    assert planner.target == target
    payload['player']['position'] = dict(zip(('x', 'y'), target))
    planner.apply(payload, risks, 3, 40)
    assert planner.arrived and planner.target is None


def test_route_cannot_cross_enemy_or_boundary():
    planner = SafeZonePlanner()
    payload = state()
    payload['enemies'] = [{'position': {'x': 610, 'y': 500}, 'radius': 30}]
    assert planner._route_score(payload, (840, 500)) is None
    assert planner._route_score(payload, (50, 500)) is None
    assert planner._route_score(payload, (280, 500)) is not None


def test_new_projectile_blocks_target_and_immediate_risk_wins():
    planner = SafeZonePlanner()
    payload = state()
    risks = {a: HazardRisk(projectile=1) for a in range(9)}
    risks[4] = HazardRisk()
    assert planner.apply(payload, risks, 4, 40) == 4
    payload['projectiles'] = [{'position': {'x': 610, 'y': 500}, 'radius': 20}]
    assert planner.apply(payload, risks, 4, 40) == 4
    assert planner.target is None
    risks[0] = HazardRisk()
    risks[4] = HazardRisk(projectile=1)
    assert planner.apply(payload, risks, 0, 40) == 0
    assert planner.target is None


def test_distant_late_wave_crowd_does_not_force_escape():
    from brotato_ai.control.recovery import CrowdRecoveryGuard
    payload = state()
    payload['wave'] = {'number': 15}
    payload['enemies'] = [{'position': {'x': 950, 'y': 500}} for _ in range(20)]
    controller = CrowdRecoveryGuard()
    controller.apply(payload, 0, risks={a: HazardRisk() for a in range(9)})
    assert not controller.active


def test_arrival_exits_escape_without_returning_unsafe_request():
    from brotato_ai.control.recovery import CrowdRecoveryGuard
    payload = state()
    controller = CrowdRecoveryGuard(hold_steps=1)
    controller.state = controller.ESCAPE
    controller._age = 2
    controller.safe_zone.target = (500, 500)
    risks = {a: HazardRisk() for a in range(9)}
    risks[4] = HazardRisk(projectile=1)
    result = controller.apply(payload, 4, risks=risks)
    assert result.applied_action == 0
    assert not controller.active


def test_wall_idle_takes_open_inward_lane():
    from brotato_ai.control.recovery import CrowdRecoveryGuard
    from brotato_ai.domain.actions import ACTION_VECTORS
    payload = state()
    payload['player']['position']['x'] = 880
    controller = CrowdRecoveryGuard()
    result = controller.apply(payload, 0, risks={a: HazardRisk() for a in range(9)})
    assert controller.active
    assert ACTION_VECTORS[result.applied_action][0] < 0
    assert controller.safe_zone.target[0] < 820


def test_short_lateral_lane_breaks_spacing_deadlock():
    payload = state()
    # Long routes fail, but there is a short, screened step upwards.
    payload['enemies'] = [{'position': {'x': 500, 'y': 250}, 'radius': 20}]
    risks = {a: HazardRisk(projectile=2) for a in range(9)}
    risks[0] = HazardRisk()
    risks[1] = HazardRisk(ranged_spacing=.2)
    planner = SafeZonePlanner()
    assert planner.apply(payload, risks, 0, 40) == 1
    assert planner.target == (500, 400)


def test_spacing_exception_does_not_relax_collision_checks():
    payload = state()
    for danger in (HazardRisk(enemy=.1), HazardRisk(enemy_path=.1),
                   HazardRisk(projectile=.1), HazardRisk(indicator=.1),
                   HazardRisk(boundary=.1)):
        risks = {a: danger for a in range(9)}
        risks[0] = HazardRisk()
        assert SafeZonePlanner().apply(payload, risks, 0, 40) == 0


def test_fast_projectile_between_samples_blocks_route():
    payload = state()
    payload['projectiles'] = [{'position': {'x': 550, 'y': 0},
                               'velocity': {'x': 0, 'y': 3000}, 'radius': 1}]
    assert SafeZonePlanner()._route_score(payload, (800, 500)) is None


def test_blocked_inward_lane_does_not_force_wall_escape():
    from brotato_ai.control.recovery import CrowdRecoveryGuard
    payload = state()
    payload['player']['position']['x'] = 880
    risks = {a: HazardRisk(projectile=2) for a in range(9)}
    risks[0] = HazardRisk()
    assert CrowdRecoveryGuard().apply(payload, 0, risks=risks).applied_action == 0
