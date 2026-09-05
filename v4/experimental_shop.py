"""Opt-in shop experiment: earlier weapon growth and diminishing stat priorities."""
from typing import Mapping, Any

from v4.ui_build_policy import RangedSmgTeacher, effect_totals


class AdaptiveSmgTeacher(RangedSmgTeacher):
    @staticmethod
    def _early_weapon_cap(wave: int) -> int:
        return 2 if wave <= 2 else 3 if wave <= 4 else 4 if wave <= 6 else 6

    def score_choice(self, choice: Mapping[str, Any], wave: int, state=None) -> float:
        score = super().score_choice(choice, wave, state)
        if score <= -120:
            return score
        state = state or {}
        stats = state.get("build", {}).get("stats", {})
        # Experimental targets, not claims of an optimal Brotato build.
        targets = {"stat_armor": min(12., 2. + wave * .6),
                   "stat_max_hp": 20. + wave * 2., "stat_speed": 10. + wave * .5,
                   "stat_lifesteal": 3. + wave * .3}
        for key, value in effect_totals(choice).items():
            if key not in targets or value <= 0:
                continue
            current = stats.get(key)
            if current is None and key == "stat_max_hp":
                current = state.get("player", {}).get("max_health")
            if current is None:
                continue
            target = targets[key]
            deficit = max(-1., min(1., (target - float(current)) / max(1., target)))
            score += self.stat_weights.get(key, 0.) * float(value) * deficit
        return score
