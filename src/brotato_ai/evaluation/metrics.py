"""Evaluation accumulators with explicit sample/window semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from brotato_ai.domain.actions import ACTION_VECTORS, MoveAction


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


@dataclass
class TacticalMetrics:
    """Replay metrics for persistent escape episodes and enemy re-entry."""

    samples: int = 0
    escape_entries: int = 0
    escape_steps: int = 0
    successful_escapes: int = 0
    post_escape_reentries: int = 0
    escape_direction_reversals: int = 0
    separation_sum: float = 0.0
    separation_samples: int = 0
    ranged_distance_sum: float = 0.0
    ranged_distance_samples: int = 0
    overrides: int = 0
    switches: int = 0
    _was_escape: bool = False
    _cleared: bool = False
    _monitor_reentry: bool = False
    _last_action: int | None = None
    _last_timestamp_ms: int | None = None
    _escape_duration_sec: float = 0.0
    _durations: list[float] = field(default_factory=list)

    def observe(
        self,
        *,
        requested_action: int,
        selected_action: int,
        escape_active: bool,
        separation: float = 0.0,
        target_distance: float = 0.0,
        ranged_active: bool = False,
        timestamp_ms: int = -1,
    ) -> None:
        self.samples += 1
        self.overrides += int(int(selected_action) != int(requested_action))
        if self._last_action is not None:
            self.switches += int(int(selected_action) != self._last_action)
            previous = ACTION_VECTORS[MoveAction(self._last_action)]
            current = ACTION_VECTORS[MoveAction(int(selected_action))]
            if previous[0] * current[0] + previous[1] * current[1] < -0.70:
                self.escape_direction_reversals += int(escape_active)
        if separation > 0.0 and math.isfinite(separation):
            self.separation_sum += float(separation)
            self.separation_samples += 1
        if ranged_active and separation > 0.0 and math.isfinite(separation):
            self.ranged_distance_sum += float(separation)
            self.ranged_distance_samples += 1
        if escape_active and not self._was_escape:
            self.escape_entries += 1
            self._escape_duration_sec = 0.0
            self._cleared = False
        if escape_active:
            self.escape_steps += 1
            if self._last_timestamp_ms is not None and timestamp_ms >= self._last_timestamp_ms:
                self._escape_duration_sec += min(1.0, max(0.0, timestamp_ms - self._last_timestamp_ms) / 1000.0)
            if target_distance > 0.0 and separation >= target_distance * 1.15:
                self._cleared = True
        elif self._was_escape:
            if self._cleared:
                self.successful_escapes += 1
                self._monitor_reentry = True
            self._durations.append(self._escape_duration_sec)
            self._escape_duration_sec = 0.0
        if (
            not escape_active
            and self._monitor_reentry
            and target_distance > 0.0
            and separation < target_distance
        ):
            self.post_escape_reentries += 1
            self._monitor_reentry = False
        self._was_escape = bool(escape_active)
        self._last_action = int(selected_action)
        self._last_timestamp_ms = int(timestamp_ms)

    def to_dict(self) -> dict[str, float | int]:
        count = max(1, self.samples)
        separation_count = max(1, self.separation_samples)
        ranged_count = max(1, self.ranged_distance_samples)
        return {
            "samples": self.samples,
            "escape_entries": self.escape_entries,
            "escape_steps": self.escape_steps,
            "successful_escapes": self.successful_escapes,
            "post_escape_reentries": self.post_escape_reentries,
            "post_escape_reentry_rate": self.post_escape_reentries / max(1, self.successful_escapes),
            "escape_direction_reversals": self.escape_direction_reversals,
            "mean_escape_duration_sec": sum(self._durations) / max(1, len(self._durations)),
            "mean_enemy_separation": self.separation_sum / separation_count,
            "mean_ranged_distance": self.ranged_distance_sum / ranged_count if self.ranged_distance_samples else 0.0,
            "ranged_distance_samples": self.ranged_distance_samples,
            "override_rate": self.overrides / count,
            "direction_switches": self.switches,
            "direction_switch_rate": self.switches / max(1, self.samples - 1),
        }
