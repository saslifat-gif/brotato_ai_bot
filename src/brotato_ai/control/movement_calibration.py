"""Ignore unverified stat-derived speed until bridge velocity is calibrated."""
import math
from brotato_ai.domain.actions import ACTION_VECTORS


def calibrate_startup_speed(state):
    combat = state.get('combat', {})
    if state.get('phase') != 'combat' or combat.get('move_speed_source') != 'stat_fallback':
        return state
    # This game's get_stat fallback has returned the same 400 for unrelated
    # stats. It is not valid evidence of a +400% movement modifier.
    result = dict(state)
    result['combat'] = dict(combat)
    result['combat']['reported_move_speed'] = combat.get('move_speed')
    result['combat']['move_speed'] = 300.
    result['combat']['move_speed_source'] = 'uncalibrated_base_speed'
    player = state.get('player', {}).get('position', {})
    arena = state.get('arena', {})
    width, height = float(arena.get('width', 0)), float(arena.get('height', 0))
    paths = dict(state.get('projectile_paths', {}))
    if width > 0 and height > 0:
        boundary = []
        for ax, ay in ACTION_VECTORS.values():
            x, y = float(player.get('x', 0))+ax*150., float(player.get('y', 0))+ay*150.
            clearance = min(x, y, width-x, height-y)
            boundary.append(max(0., min(1., (80.-clearance)/80.)))
        paths['boundary_action_risk'] = boundary
        result['projectile_paths'] = paths
    return result
