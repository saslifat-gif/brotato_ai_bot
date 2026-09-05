"""Bounded attraction toward nearby materials, using existing hazard estimates."""
import math
from collections.abc import Mapping

from brotato_ai.domain.actions import ACTION_VECTORS


def material_progress(payload):
    player = payload.get('player', {})
    if payload.get('phase') != 'combat' or payload.get('dead') or payload.get('victory'):
        return {}
    if float(player.get('health', 0)) / max(1., float(player.get('max_health', 1))) < .35:
        return {}  # Leave low-health healing/escape decisions alone.
    position = player.get('position', {})
    px, py = float(position.get('x', 0)), float(position.get('y', 0))
    targets = []
    for item in payload.get('pickups', []):
        if not isinstance(item, Mapping):
            continue
        if item.get('category', item.get('kind')) != 'material':
            continue
        pos = item.get('position', {})
        dx, dy = float(pos.get('x', 0)) - px, float(pos.get('y', 0)) - py
        distance = math.hypot(dx, dy)
        if not 12 < distance <= 450:
            continue
        value = max(1., min(10., float(item.get('material_value', 1))))
        targets.append((distance, dx, dy, math.sqrt(value)))
    targets.sort()
    targets = targets[:24]
    if not targets:
        return {}
    scores = {}
    for action, (ax, ay) in ACTION_VECTORS.items():
        scores[int(action)] = sum(
            weight * (distance - math.hypot(dx - 60 * ax, dy - 60 * ay))
            / (60 * (1 + distance / 150))
            for distance, dx, dy, weight in targets
        )
    scale = max(1., max(abs(s) for s in scores.values()))
    return {a: s / scale for a, s in scores.items()}


def prefer_materials(payload, risks, current):
    progress = material_progress(payload)
    if not progress:
        return current
    base = risks[current]
    candidates = [a for a, risk in risks.items()
                  if risk.total <= min(.20, base.total + .03)
                  and risk.enemy_total <= base.enemy_total + .02
                  and risk.projectile_total <= base.projectile_total + .02
                  and risk.indicator <= base.indicator + .02
                  and risk.boundary_total <= base.boundary_total + .02]
    if not candidates:
        return current
    best = max(candidates, key=lambda a: (progress[a], a == current, -a))
    return best if progress[best] > max(0., progress[current]) + .15 else current
