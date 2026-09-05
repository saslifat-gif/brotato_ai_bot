"""Short, persistent escape destinations with conservative route screening."""
import math
from brotato_ai.domain.actions import ACTION_VECTORS


def xy(item):
    pos = item.get('position', {})
    return float(pos.get('x', 0)), float(pos.get('y', 0))


class SafeZonePlanner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.age = 0.
        self.stalled = 0.
        self.last_distance = None
        self.arrived = False

    def _route_score(self, payload, target):
        px, py = xy(payload.get('player', {}))
        tx, ty = target
        arena = payload.get('arena', {})
        width, height = float(arena.get('width', 0)), float(arena.get('height', 0))
        margin = 100.
        if not margin <= tx <= width-margin or not margin <= ty <= height-margin:
            return None
        distance = math.hypot(tx-px, ty-py)
        speed = max(150., float(payload.get('combat', {}).get('move_speed', 300)))
        travel = min(1.2, distance/speed)
        pressure = 0.
        for kind in ('enemies', 'projectiles', 'attack_indicators'):
            for item in payload.get(kind, ()):
                if item.get('dead'):
                    continue
                ex, ey = xy(item)
                velocity = item.get('velocity', {})
                vx, vy = float(velocity.get('x', 0)), float(velocity.get('y', 0))
                radius = max(12., float(item.get('radius', 40))) + 48.
                start_clearance = math.hypot(ex-px, ey-py)-radius
                for fraction in (.25, .5, .75, 1.):
                    clearance = math.hypot(ex+vx*travel*fraction-(px+(tx-px)*fraction),
                                           ey+vy*travel*fraction-(py+(ty-py)*fraction))-radius
                    # Permit exiting an existing overlap, never crossing into one.
                    if clearance < min(20., start_clearance+10.):
                        return None
                    pressure += max(0., 220.-clearance)/220. * (.5 if fraction < 1 else 1.)
        edge = min(tx, ty, width-tx, height-ty)
        pressure += max(0., 180.-edge)/180.
        # Coins distinguish otherwise similarly open destinations.
        coins = sum(1 for p in payload.get('pickups', ())
                    if p.get('category', p.get('kind')) == 'material'
                    and math.hypot(xy(p)[0]-tx, xy(p)[1]-ty) < 120.)
        return pressure - min(.12, coins*.02) + distance/10000.

    def apply(self, payload, risks, fallback, interval_ms):
        self.arrived = False
        dt = min(.2, max(0., interval_ms)/1000.) or 1/24
        px, py = xy(payload.get('player', {}))
        self.age += dt
        if self.target is not None:
            distance = math.hypot(self.target[0]-px, self.target[1]-py)
            if distance <= 65.:
                self.arrived = True
                self.target = None
                return fallback
            self.stalled = self.stalled+dt if self.last_distance is not None and distance >= self.last_distance-1 else 0.
            self.last_distance = distance
            if self.age > 1.5 or self.stalled > .5 or self._route_score(payload, self.target) is None:
                self.target = None
        base = risks[fallback]
        allowed = [a for a, r in risks.items() if a != 0
                   and r.total <= base.total+.03
                   and r.enemy_total <= base.enemy_total+.02
                   and r.projectile_total <= base.projectile_total+.02
                   and r.indicator <= base.indicator+.02
                   and r.boundary_total <= base.boundary_total+.02]
        if not allowed:
            self.target = None
            return fallback
        if self.target is None:
            options = []
            for action in allowed:
                ax, ay = ACTION_VECTORS[action]
                for length in (220., 340.):
                    target = (px+ax*length, py+ay*length)
                    score = self._route_score(payload, target)
                    if score is not None:
                        options.append((score, action, target))
            if not options:
                return fallback
            self.target = min(options)[2]
            self.age = self.stalled = 0.
            self.last_distance = None
        dx, dy = self.target[0]-px, self.target[1]-py
        distance = math.hypot(dx, dy)
        best = max(allowed, key=lambda a: (dx*ACTION_VECTORS[a][0]+dy*ACTION_VECTORS[a][1], a == fallback))
        if dx*ACTION_VECTORS[best][0]+dy*ACTION_VECTORS[best][1] < distance*.3:
            self.target = None
            return fallback
        return best
