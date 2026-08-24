"""Reward calculation from exact bridge state instead of screen pixels."""

from typing import Any, Mapping, Optional


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _nested(state: Mapping[str, Any], group: str, key: str, default: float = 0.0) -> float:
    value = state.get(group, {})
    return _number(value.get(key), default) if isinstance(value, Mapping) else float(default)


class ApiRewardEngine:
    def __init__(self):
        self.previous: Optional[Mapping[str, Any]] = None

    def reset(self, state: Mapping[str, Any]) -> None:
        self.previous = state

    def step(self, state: Mapping[str, Any]) -> float:
        wave_number = max(0.0, _nested(state, "wave", "number"))
        # Later waves are the curriculum bottleneck. Keep the shaping modest
        # early, then make survival/completion increasingly important from the
        # middle of a run onward.
        late_wave_scale = 1.0 + min(20.0, wave_number) * 0.05
        reward = 0.01 * late_wave_scale if state.get("phase") == "combat" else 0.0
        previous = self.previous
        if previous is not None:
            old_max = max(1.0, _nested(previous, "player", "max_health", 1.0))
            new_max = max(1.0, _nested(state, "player", "max_health", old_max))
            old_hp = _nested(previous, "player", "health") / old_max
            new_hp = _nested(state, "player", "health") / new_max
            hp_delta = new_hp - old_hp
            reward += hp_delta * (2.0 if hp_delta > 0 else 10.0)

            materials_delta = _nested(state, "counters", "materials") - _nested(
                previous, "counters", "materials"
            )
            reward += max(0.0, min(10.0, materials_delta)) * 0.05
            kills_delta = _nested(state, "counters", "kills") - _nested(
                previous, "counters", "kills"
            )
            reward += max(0.0, min(10.0, kills_delta)) * 0.5
            wave_delta = _nested(state, "wave", "number") - _nested(previous, "wave", "number")
            reward += max(0.0, wave_delta) * (10.0 + min(20.0, wave_number) * 2.0)
            if (
                previous.get("phase") == "combat"
                and state.get("phase") == "wave_end"
                and not state.get("dead")
            ):
                reward += 10.0
        if state.get("dead"):
            reward -= 25.0 + min(20.0, wave_number) * 2.0
        if state.get("victory"):
            reward += 100.0
        self.previous = state
        return float(reward)
