"""Evaluation accumulators with explicit sample/window semantics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VariantMetrics:
    samples: int = 0
    risk_sum: float = 0.0
    requested_risk_sum: float = 0.0
    overrides: int = 0
    minimum_risk_actions: int = 0
    unsafe_action_count: int = 0
    unsafe_action_total: int = 0
    requested_regret_sum: float = 0.0
    direction_switches: int = 0
    previous_action: int | None = None

    def observe(
        self, *, requested_action: int, selected_action: int, requested_risk: float, selected_risk: float, minimum_action: int, unsafe_action_count: int = 0, action_count: int = 9, minimum_risk: float = 0.0
    ) -> None:
        self.samples += 1
        self.risk_sum += float(selected_risk)
        self.requested_risk_sum += float(requested_risk)
        self.overrides += int(selected_action != requested_action)
        self.minimum_risk_actions += int(selected_action == minimum_action)
        self.unsafe_action_count += int(unsafe_action_count)
        self.unsafe_action_total += int(action_count)
        self.requested_regret_sum += max(0.0, float(requested_risk) - float(minimum_risk))
        if self.previous_action is not None:
            self.direction_switches += int(selected_action != self.previous_action)
        self.previous_action = selected_action

    def to_dict(self) -> dict[str, float | int]:
        count = max(1, self.samples)
        return {
            "samples": self.samples,
            "mean_modeled_risk": self.risk_sum / count,
            "mean_requested_risk": self.requested_risk_sum / count,
            "risk_reduction": (self.requested_risk_sum - self.risk_sum) / count,
            "override_rate": self.overrides / count,
            "minimum_risk_action_rate": self.minimum_risk_actions / count,
            "mean_unsafe_action_count": self.unsafe_action_count / count,
            "mean_unsafe_action_fraction": self.unsafe_action_count / max(1, self.unsafe_action_total),
            "mean_requested_to_minimum_regret": self.requested_regret_sum / count,
            "direction_switches": self.direction_switches,
            "direction_switch_rate": self.direction_switches / max(1, self.samples - 1),
        }


@dataclass
class DamageMetrics:
    damage_samples: int = 0
    unique_damage_windows: int = 0
    total_damage: float = 0.0
    deaths: int = 0
    victories: int = 0
    maximum_wave: int = 0
    _last_session: str = ""
    _last_health: float | None = None
    _last_damage_ms: int | None = None
    _last_dead: bool = False
    _last_victory: bool = False

    def observe(
        self,
        *,
        session: str,
        timestamp_ms: int,
        health: float,
        wave: int,
        dead: bool,
        victory: bool,
        unique_window_ms: int = 500,
    ) -> None:
        self.maximum_wave = max(self.maximum_wave, int(wave))
        if session != self._last_session:
            self._last_session = session
            self._last_health = health
            self._last_damage_ms = None
            self._last_dead = False
            self._last_victory = False
        elif self._last_health is not None and health < self._last_health:
            damage = self._last_health - health
            self.damage_samples += 1
            self.total_damage += damage
            if (
                self._last_damage_ms is None
                or timestamp_ms < self._last_damage_ms
                or timestamp_ms - self._last_damage_ms > unique_window_ms
            ):
                self.unique_damage_windows += 1
            self._last_damage_ms = timestamp_ms
        self._last_health = health
        self.deaths += int(dead and not self._last_dead)
        self.victories += int(victory and not self._last_victory)
        self._last_dead = bool(dead)
        self._last_victory = bool(victory)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "damage_samples": self.damage_samples,
            "unique_damage_windows": self.unique_damage_windows,
            "total_damage": self.total_damage,
            "death_events": self.deaths,
            "victory_events": self.victories,
            "maximum_wave": self.maximum_wave,
        }
