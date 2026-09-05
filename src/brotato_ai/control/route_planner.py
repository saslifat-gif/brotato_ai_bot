"""Experimental two-stage route proposals; the existing arbiter still writes actions."""
from copy import deepcopy
from typing import Mapping, Any

from brotato_ai.control.hazards import UnifiedHazardScorer
from brotato_ai.domain.actions import ACTION_VECTORS


class EscapeRoutePlanner:
    def __init__(self, horizon_seconds: float = .45, improvement_margin: float = .15):
        self.horizon_seconds = horizon_seconds
        self.improvement_margin = improvement_margin
        self.scorer = UnifiedHazardScorer()

    def forecast(self, state: Mapping[str, Any], action: int) -> dict:
        future = deepcopy(dict(state))
        player = future.setdefault("player", {})
        pos = player.setdefault("position", {"x": 0., "y": 0.})
        speed = max(1., float(future.get("combat", {}).get("move_speed", 300)))
        dx, dy = ACTION_VECTORS[action]
        pos["x"] += dx * speed * self.horizon_seconds
        pos["y"] += dy * speed * self.horizon_seconds
        player["velocity"] = {"x": dx * speed, "y": dy * speed}
        for group in ("enemies", "projectiles"):
            for entity in future.get(group, []):
                position, velocity = entity.get("position", {}), entity.get("velocity", {})
                position["x"] = float(position.get("x", 0)) + float(velocity.get("x", 0)) * self.horizon_seconds
                position["y"] = float(position.get("y", 0)) + float(velocity.get("y", 0)) * self.horizon_seconds
        for indicator in future.get("attack_indicators", []):
            if "time_to_activate" in indicator:
                indicator["time_to_activate"] = max(0., float(indicator["time_to_activate"]) - self.horizon_seconds)
        # These grids describe the actual state, not the forecast location.
        future.pop("projectile_paths", None)
        future.pop("arena_grid", None)
        return future

    def propose(self, state: Mapping[str, Any], requested: int) -> tuple[int, dict]:
        risks = self.scorer.all_risks(state)
        requested_risk = risks[requested].total
        candidates = sorted(risks, key=lambda a: (risks[a].total, a))
        candidates = [a for a in candidates if risks[a].total <= requested_risk + .08][:3]
        candidates = list(dict.fromkeys([requested, *candidates]))
        scores, exits = {}, {}
        for action in candidates:
            later = self.scorer.all_risks(self.forecast(state, action))
            exits[action] = sum(r.total < .65 for a, r in later.items() if a != 0)
            scores[action] = risks[action].total + .5 * min(r.total for r in later.values()) + .025 * (8 - exits[action])
        best = min(scores, key=lambda a: (scores[a], a != requested, a))
        chosen = best if scores[best] + self.improvement_margin < scores[requested] else requested
        return chosen, {"original_action": requested, "proposed_action": chosen,
                        "scores": scores, "future_exits": exits, "horizon_seconds": 2 * self.horizon_seconds}
