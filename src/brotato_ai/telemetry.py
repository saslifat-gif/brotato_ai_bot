"""Small, dependency-free helpers for runtime and replay telemetry."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return a deterministic percentile, or 0 for an empty/non-finite sample."""
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    q = min(1.0, max(0.0, float(quantile)))
    if len(finite) == 1:
        return finite[0]
    position = q * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = min(len(finite) - 1, lower + 1)
    fraction = position - lower
    return finite[lower] + (finite[upper] - finite[lower]) * fraction


def reward_time_scale(interval_ms: float, reference_hz: float) -> float:
    """Scale dense rewards by elapsed reference-time, with safety bounds."""
    hz = max(1.0, float(reference_hz))
    interval = float(interval_ms)
    if not math.isfinite(interval) or interval <= 0.0:
        interval = 1000.0 / hz
    return min(4.0, max(0.25, interval * hz / 1000.0))


def risk_diagnostics(
    risks: Mapping[int, object],
    requested_action: int,
    *,
    unsafe_threshold: float = 0.65,
) -> dict[str, float | int]:
    """Summarize all-action risk without treating it as collision probability."""
    values = {}
    for action, risk in risks.items():
        try:
            values[int(action)] = float(getattr(risk, "total"))
        except (TypeError, ValueError):
            continue
    if not values:
        return {
            "minimum_action_risk": 0.0,
            "unsafe_action_count": 0,
            "unsafe_action_fraction": 0.0,
            "requested_to_minimum_regret": 0.0,
        }
    minimum = min(values.values())
    requested = values.get(int(requested_action), minimum)
    unsafe = sum(value >= float(unsafe_threshold) for value in values.values())
    return {
        "minimum_action_risk": minimum,
        "unsafe_action_count": int(unsafe),
        "unsafe_action_fraction": unsafe / max(1, len(values)),
        "requested_to_minimum_regret": max(0.0, requested - minimum),
    }
