"""Conservative short-term clearance evidence from observed enemy damage."""
from collections import deque
from dataclasses import replace
import math

from brotato_ai.control.hazards import ranged_spacing_risk


class KillForecast:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = {}
        self.context = None
        self.last_time = None
        self.clearable_count = 0

    def update(self, payload, risks):
        self.clearable_count = 0
        now = float(payload.get('timestamp_ms', -1)) / 1000.
        context = (payload.get('session'), str(payload.get('wave')))
        if (payload.get('phase') != 'combat' or payload.get('dead') or payload.get('victory')
                or not math.isfinite(now) or now < 0):
            self.reset()
            return payload, risks
        if context != self.context or (self.last_time is not None and now < self.last_time):
            self.reset()
        self.context = context
        if self.last_time is not None and now <= self.last_time:
            return payload, risks
        self.last_time = now
        player = payload.get('player', {})
        pos = player.get('position', {})
        combat = payload.get('combat', {})
        speed = float(combat.get('move_speed', 0))
        weapon_range = float(combat.get('weapon_range', 0))
        ranged = float(combat.get('ranged_count', 0)) > float(combat.get('melee_count', 0))
        alive = set()
        clearable = set()
        for enemy in payload.get('enemies', ()):
            key = enemy.get('runtime_id')
            hp = float(enemy.get('health', -1))
            if not key or not math.isfinite(hp) or hp <= 0:
                continue
            alive.add(key)
            history = self.history.setdefault(key, deque(maxlen=32))
            if history and (hp > history[-1][1] or now-history[-1][0] > .20):
                history.clear()
            history.append((now, hp))
            while history and now-history[0][0] > .65:
                history.popleft()
            hits = [(t, old-h) for (_, old), (t, h) in zip(history, list(history)[1:]) if old > h]
            if (len(hits) < 3 or now-hits[-1][0] > .15 or not ranged or speed <= 0
                    or enemy.get('is_boss') or enemy.get('is_elite') or enemy.get('is_charging')):
                continue
            # Require sustained damage, use the weakest hit and longest interval,
            # and halve that rate to allow for misses or a target switch.
            gaps = [b[0]-a[0] for a, b in zip(hits, hits[1:])]
            interval = max(gaps)
            if interval <= 0 or interval > .20:
                continue
            dps = .5 * min(hit[1] for hit in hits) / interval
            ttk = hp / dps + interval
            ep = enemy.get('position', {})
            distance = math.hypot(float(ep.get('x', 0))-float(pos.get('x', 0)),
                                  float(ep.get('y', 0))-float(pos.get('y', 0)))
            velocity = enemy.get('velocity', {})
            enemy_speed = math.hypot(float(velocity.get('x', 0)), float(velocity.get('y', 0)))
            radius = max(28., float(player.get('radius', 28)), float(player.get('width', 0))/2,
                         float(player.get('height', 0))/2) + max(40., float(enemy.get('radius', 40)))
            earliest_contact = max(0., distance-radius) / (speed+enemy_speed)
            if ttk <= .45 and ttk+.20 < earliest_contact and distance < weapon_range*.85:
                clearable.add(key)
        self.history = {key: value for key, value in self.history.items() if key in alive}
        if not clearable:
            return payload, risks
        filtered = dict(payload)
        filtered['enemies'] = [e for e in payload.get('enemies', ()) if e.get('runtime_id') not in clearable]
        self.clearable_count = len(clearable)
        adjusted = {}
        for action, risk in risks.items():
            old = ranged_spacing_risk(payload, action)
            new = ranged_spacing_risk(filtered, action)
            # Preserve the scorer's configured scale, and never increase a risk.
            spacing = risk.ranged_spacing * min(1., new/old) if old > 0 else risk.ranged_spacing
            adjusted[action] = replace(risk, ranged_spacing=spacing)
        return filtered, adjusted
