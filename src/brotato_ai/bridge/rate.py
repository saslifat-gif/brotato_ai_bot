"""Measured state and control rates; never confuse these with training FPS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateSample:
    interval_ms: float = 0.0
    effective_hz: float = 0.0


class RateMeter:
    def __init__(self, smoothing: float = 0.10):
        self.smoothing = min(1.0, max(0.001, float(smoothing)))
        self.previous_ms: float | None = None
        self.effective_hz = 0.0
        self.last_interval_ms = 0.0

    def reset(self) -> None:
        self.previous_ms = None
        self.effective_hz = 0.0
        self.last_interval_ms = 0.0

    def observe(self, timestamp_ms: float) -> RateSample:
        current = float(timestamp_ms)
        previous = self.previous_ms
        self.previous_ms = current
        if previous is None or current <= previous:
            return RateSample(self.last_interval_ms, self.effective_hz)
        self.last_interval_ms = current - previous
        instantaneous = 1000.0 / max(0.001, self.last_interval_ms)
        if self.effective_hz <= 0.0:
            self.effective_hz = instantaneous
        else:
            alpha = self.smoothing
            self.effective_hz = (1.0 - alpha) * self.effective_hz + alpha * instantaneous
        return RateSample(self.last_interval_ms, self.effective_hz)

