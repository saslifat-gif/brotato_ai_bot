import pytest
from brotato_ai.control.recovery import CrowdRecoveryGuard
from brotato_ai.domain.decisions import HazardRisk


def frame():
    return {'phase': 'combat', 'wave': {'number': 17},
            'arena': {'width': 1000, 'height': 600},
            'player': {'position': {'x': 500, 'y': 300}, 'health': 50, 'max_health': 100},
            'combat': {'move_speed': 300},
            'enemies': [{'position': {'x': 600, 'y': 300}, 'velocity': {'x': -50, 'y': 0}, 'radius': 40}]}


def risks():
    result = {a: HazardRisk(projectile=3) for a in range(9)}
    result[0] = HazardRisk(enemy=.3)
    result[3] = HazardRisk(enemy=.38, ranged_spacing=.08)
    return result


def test_repeated_escape_idle_becomes_bounded_away_move():
    guard = CrowdRecoveryGuard()
    choices = [guard.apply(frame(), 0, risks=risks(), control_interval_ms=100).applied_action for _ in range(4)]
    assert choices == [0, 0, 0, 3]
    assert guard.anti_stall_active
    assert guard.idle_escape_ms == 0


@pytest.mark.parametrize('danger', [HazardRisk(projectile=.5), HazardRisk(boundary=.5), HazardRisk(indicator=.5), HazardRisk(enemy=2)])
def test_stall_never_forces_a_hazardous_escape(danger):
    guard = CrowdRecoveryGuard()
    assessment = risks()
    assessment[3] = danger
    assert all(guard.apply(frame(), 0, risks=assessment, control_interval_ms=100).applied_action == 0 for _ in range(10))
    assert not guard.anti_stall_active


def test_safe_idle_and_episode_reset_do_not_trigger_escape():
    guard = CrowdRecoveryGuard()
    assessment = risks()
    assessment[0] = HazardRisk()
    for _ in range(10):
        assert guard._break_dangerous_idle(frame(), assessment, 0, 100) == 0
    guard.reset()
    assert guard.idle_escape_ms == 0
    assert not guard.anti_stall_active


def test_stationary_timer_uses_elapsed_time_not_step_count():
    for interval in (25, 50, 100):
        guard = CrowdRecoveryGuard()
        elapsed = 0
        while True:
            elapsed += interval
            result = guard._break_dangerous_idle(frame(), risks(), 0, interval)
            if result:
                break
            assert elapsed < 350
        assert 350 <= elapsed < 350 + interval
