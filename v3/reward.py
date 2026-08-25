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
    """Outcome-dominant reward for combat and structured menu transitions.

    The previous reward made kills and materials too valuable relative to
    surviving a wave.  This version keeps those signals as small shaping terms
    while making wave completion, death, and victory the dominant objectives.
    ``last_components`` is exposed for TensorBoard diagnostics and does not
    change the public ``step() -> float`` API.
    """

    SURVIVAL_REWARD = 0.02
    POSITIVE_HEALTH_SCALE = 2.0
    NEGATIVE_HEALTH_SCALE = 8.0
    MATERIAL_REWARD = 0.015
    KILL_REWARD = 0.10
    WAVE_ADVANCE_BASE = 25.0
    WAVE_ADVANCE_SCALE = 2.0
    WAVE_CLEAR_REWARD = 30.0
    DEATH_PENALTY_BASE = 100.0
    DEATH_PENALTY_SCALE = 4.0
    VICTORY_REWARD = 300.0

    def __init__(self):
        self.previous: Optional[Mapping[str, Any]] = None
        self.last_components: dict[str, float] = {}

    def reset(self, state: Mapping[str, Any]) -> None:
        self.previous = state
        self.last_components = {}

    def step(self, state: Mapping[str, Any]) -> float:
        wave_number = max(0.0, _nested(state, "wave", "number"))
        # Survival is deliberately small but present at every combat sample;
        # terminal outcomes and wave completion remain much larger.
        late_wave_scale = 1.0 + min(20.0, wave_number) * 0.05
        components = {
            "survival": self.SURVIVAL_REWARD * late_wave_scale
            if state.get("phase") == "combat"
            else 0.0,
            "health": 0.0,
            "kills": 0.0,
            "materials": 0.0,
            "wave_advance": 0.0,
            "wave_clear": 0.0,
            "death": 0.0,
            "victory": 0.0,
        }
        previous = self.previous
        if previous is not None:
            old_max = max(1.0, _nested(previous, "player", "max_health", 1.0))
            new_max = max(1.0, _nested(state, "player", "max_health", old_max))
            old_hp = _nested(previous, "player", "health") / old_max
            new_hp = _nested(state, "player", "health") / new_max
            hp_delta = new_hp - old_hp
            components["health"] += hp_delta * (
                self.POSITIVE_HEALTH_SCALE
                if hp_delta > 0
                else self.NEGATIVE_HEALTH_SCALE
            )

            materials_delta = _nested(state, "counters", "materials") - _nested(
                previous, "counters", "materials"
            )
            components["materials"] += max(0.0, min(5.0, materials_delta)) * self.MATERIAL_REWARD
            kills_delta = _nested(state, "counters", "kills") - _nested(
                previous, "counters", "kills"
            )
            components["kills"] += max(0.0, min(4.0, kills_delta)) * self.KILL_REWARD
            wave_delta = _nested(state, "wave", "number") - _nested(previous, "wave", "number")
            components["wave_advance"] += max(0.0, wave_delta) * (
                self.WAVE_ADVANCE_BASE
                + min(20.0, wave_number) * self.WAVE_ADVANCE_SCALE
            )
            if (
                previous.get("phase") == "combat"
                and state.get("phase") == "wave_end"
                and not state.get("dead")
            ):
                components["wave_clear"] += self.WAVE_CLEAR_REWARD
        if state.get("dead"):
            components["death"] -= self.DEATH_PENALTY_BASE + min(20.0, wave_number) * self.DEATH_PENALTY_SCALE
        if state.get("victory"):
            components["victory"] += self.VICTORY_REWARD
        self.previous = state
        self.last_components = components
        return float(sum(components.values()))
