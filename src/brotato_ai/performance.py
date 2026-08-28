"""Low-overhead runtime timing for the live control path.

The profiler is deliberately opt-in.  When disabled, ``begin`` returns zero
and ``end`` does no clock read, so production runs do not pay for timing.  A
bounded sample reservoir keeps long sessions from turning the profiler into a
second memory or latency problem.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index] / 1_000_000.0


def _plain_percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass
class _Stage:
    calls: int = 0
    total_ns: int = 0
    minimum_ns: int | None = None
    maximum_ns: int = 0
    samples_ns: list[int] = field(default_factory=list)

    def add(self, duration_ns: int, sample_limit: int) -> None:
        duration_ns = max(0, int(duration_ns))
        self.calls += 1
        self.total_ns += duration_ns
        self.minimum_ns = duration_ns if self.minimum_ns is None else min(self.minimum_ns, duration_ns)
        self.maximum_ns = max(self.maximum_ns, duration_ns)
        if len(self.samples_ns) < sample_limit:
            self.samples_ns.append(duration_ns)
        elif sample_limit > 0:
            # Deterministic bounded reservoir sampling.  It retains tails while
            # avoiding an allocation per tick once the reservoir is full.
            index = (
                (self.calls * 1_103_515_245 + 12_345) & 0x7FFFFFFF
            ) % self.calls
            if index < sample_limit:
                self.samples_ns[index] = duration_ns


class RuntimeProfiler:
    """Collect stage timings and source/processing counters for one runtime."""

    def __init__(self, *, enabled: bool = False, sample_limit: int = 20_000):
        self.enabled = bool(enabled)
        self.sample_limit = max(100, int(sample_limit))
        self.started_ns = time.perf_counter_ns()
        self.stages: dict[str, _Stage] = {}
        self.counters: dict[str, int] = {}
        self.values: dict[str, list[float]] = {}
        self.source_samples: list[dict[str, Any]] = []
        self._source_last_timestamp: dict[tuple[str, str], int] = {}
        self._source_first_timestamp: dict[tuple[str, str], int] = {}
        self._source_counts: dict[tuple[str, str], dict[str, int]] = {}
        self._source_interval_samples: dict[tuple[str, str], list[int]] = {}
        self._source_clock_offset_ns: dict[tuple[str, str], int] = {}

    @classmethod
    def disabled(cls) -> "RuntimeProfiler":
        return cls(enabled=False)

    def begin(self, _stage: str) -> int:
        return time.perf_counter_ns() if self.enabled else 0

    def end(self, stage: str, started_ns: int) -> None:
        if not self.enabled or not started_ns:
            return
        self.stages.setdefault(stage, _Stage()).add(
            time.perf_counter_ns() - started_ns, self.sample_limit
        )

    def observe(self, stage: str, duration_ns: int) -> None:
        if self.enabled:
            self.stages.setdefault(stage, _Stage()).add(duration_ns, self.sample_limit)

    def count(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + int(amount)

    def value(self, name: str, value: float) -> None:
        if self.enabled and math.isfinite(float(value)):
            samples = self.values.setdefault(name, [])
            if len(samples) < self.sample_limit:
                samples.append(float(value))

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def source_boundary(
        self,
        *,
        receive_call_start_ns: int,
        recv_start_ns: int | None,
        response_arrival_ns: int,
        payload_complete_ns: int,
        parse_start_ns: int,
        parse_end_ns: int,
        payload_size_bytes: int,
        message: dict[str, Any],
    ) -> int | None:
        """Record one state at the transport/source boundary.

        The timestamps are monotonic local-clock values.  ``source_timestamp_ms``
        is copied from the game and is intentionally kept in its own clock
        domain so source cadence and transport cadence cannot be conflated.
        """

        if not self.enabled or message.get("type") not in {"state", "raw_state"}:
            return None
        source_timestamp = next(
            (
                self._optional_int(message.get(key))
                for key in ("published_at_ms", "game_timestamp_ms", "timestamp_ms")
                if self._optional_int(message.get(key)) is not None
            ),
            None,
        )
        source_sequence = next(
            (
                self._optional_int(message.get(key))
                for key in ("source_sequence", "raw_sequence", "sequence")
                if self._optional_int(message.get(key)) is not None
            ),
            None,
        )
        source_tick = self._optional_int(message.get("tick"))
        source_key = (str(message.get("type", "")), str(message.get("session", "")))
        counts = self._source_counts.setdefault(
            source_key, {"received": 0, "fresh": 0, "duplicates_or_old": 0}
        )
        counts["received"] += 1
        if source_timestamp is not None:
            previous_timestamp = self._source_last_timestamp.get(source_key)
            if source_key not in self._source_first_timestamp:
                self._source_first_timestamp[source_key] = source_timestamp
                self._source_clock_offset_ns[source_key] = (
                    int(response_arrival_ns) - int(source_timestamp) * 1_000_000
                )
            if previous_timestamp is None or source_timestamp > previous_timestamp:
                counts["fresh"] += 1
                if previous_timestamp is not None:
                    interval = source_timestamp - previous_timestamp
                    samples = self._source_interval_samples.setdefault(source_key, [])
                    if len(samples) < self.sample_limit:
                        samples.append(interval)
                    elif samples:
                        index = ((counts["fresh"] * 1_103_515_245 + 12_345) & 0x7FFFFFFF) % counts["fresh"]
                        if index < self.sample_limit:
                            samples[index] = interval
                self._source_last_timestamp[source_key] = source_timestamp
            else:
                counts["duplicates_or_old"] += 1
        sample = {
            "message_type": str(message.get("type", "")),
            "session": str(message.get("session", "")),
            "local_receive_time_ns": int(response_arrival_ns),
            "receive_call_start_ns": int(receive_call_start_ns),
            "recv_start_ns": int(recv_start_ns) if recv_start_ns is not None else None,
            "payload_complete_ns": int(payload_complete_ns),
            "parse_start_ns": int(parse_start_ns),
            "parse_end_ns": int(parse_end_ns),
            "receive_wait_ms": (
                (response_arrival_ns - recv_start_ns) / 1_000_000.0
                if recv_start_ns is not None
                else None
            ),
            "payload_completion_delay_ms": (
                (payload_complete_ns - response_arrival_ns) / 1_000_000.0
            ),
            "parse_ms": (parse_end_ns - parse_start_ns) / 1_000_000.0,
            "receive_to_parse_end_ms": (
                (parse_end_ns - receive_call_start_ns) / 1_000_000.0
            ),
            "source_timestamp_ms": source_timestamp,
            "source_sequence": source_sequence,
            "source_tick": source_tick,
            "bridge_eligible_at_ms": self._optional_int(message.get("bridge_eligible_at_ms")),
            "bridge_dispatch_at_ms": self._optional_int(message.get("bridge_dispatch_at_ms")),
            "payload_size_bytes": max(0, int(payload_size_bytes)),
            "enemy_count": len(message.get("enemies", []))
            if isinstance(message.get("enemies"), (list, tuple))
            else 0,
            "projectile_count": len(message.get("projectiles", []))
            if isinstance(message.get("projectiles"), (list, tuple))
            else 0,
            "processing_start_ns": None,
            "processing_end_ns": None,
            "action_decision_ns": None,
            "action_sent_ns": None,
        }
        if len(self.source_samples) >= self.sample_limit:
            # Keep the same bounded-reservoir behavior as stage samples.  The
            # newest sample replaces an older one deterministically when it is
            # selected, preserving useful tails without unbounded growth.
            index = ((len(self.source_samples) + 1) * 1_103_515_245 + 12_345) & 0x7FFFFFFF
            index %= len(self.source_samples) + 1
            if index < len(self.source_samples):
                self.source_samples[index] = sample
            return index if index < len(self.source_samples) else None
        self.source_samples.append(sample)
        self.count("source_boundary_states")
        return len(self.source_samples) - 1

    def update_source_boundary(self, index: int | None, field: str, timestamp_ns: int) -> None:
        if not self.enabled or index is None:
            return
        if not 0 <= int(index) < len(self.source_samples):
            return
        if field not in {
            "processing_start_ns",
            "processing_end_ns",
            "action_decision_ns",
            "action_sent_ns",
        }:
            raise ValueError(f"unsupported source boundary field: {field}")
        sample = self.source_samples[int(index)]
        timestamp_ns = int(timestamp_ns)
        sample[field] = timestamp_ns
        received_ns = sample.get("local_receive_time_ns")
        if isinstance(received_ns, int):
            sample[field.replace("_ns", "_after_receive_ms")] = (
                timestamp_ns - received_ns
            ) / 1_000_000.0
        source_timestamp = sample.get("source_timestamp_ms")
        source_key = (
            str(sample.get("message_type", "")),
            str(sample.get("session", "")),
        )
        offset_ns = self._source_clock_offset_ns.get(source_key)
        if isinstance(source_timestamp, int) and offset_ns is not None:
            sample_age = (
                timestamp_ns
                - (int(source_timestamp) * 1_000_000 + offset_ns)
            ) / 1_000_000.0
            if field == "action_decision_ns":
                sample["state_age_at_decision_ms_estimated"] = sample_age
                self.value("state_age_at_decision_ms", sample_age)
            elif field == "action_sent_ns":
                sample["state_age_at_action_sent_ms_estimated"] = sample_age
                self.value("state_age_at_action_sent_ms", sample_age)

    def report(self, *, wall_seconds: float | None = None) -> dict[str, Any]:
        elapsed = max(1e-9, float(wall_seconds) if wall_seconds is not None else (
            time.perf_counter_ns() - self.started_ns
        ) / 1_000_000_000.0)
        total = self.stages.get("control_loop_total")
        total_ns = total.total_ns if total else 0
        stages: dict[str, Any] = {}
        for name, stage in sorted(self.stages.items()):
            stages[name] = {
                "calls": stage.calls,
                "mean_ms": stage.total_ns / max(1, stage.calls) / 1_000_000.0,
                "p50_ms": _percentile(stage.samples_ns, 0.50),
                "p90_ms": _percentile(stage.samples_ns, 0.90),
                "p95_ms": _percentile(stage.samples_ns, 0.95),
                "p99_ms": _percentile(stage.samples_ns, 0.99),
                "maximum_ms": stage.maximum_ns / 1_000_000.0,
                "calls_per_second": stage.calls / elapsed,
                "percent_of_control_loop": (
                    100.0 * stage.total_ns / total_ns if total_ns else None
                ),
            }
        values = {}
        for name, samples in sorted(self.values.items()):
            values[name] = {
                "samples": len(samples),
                "mean": sum(samples) / len(samples) if samples else None,
                "p50": _percentile([int(value * 1_000_000) for value in samples], 0.50),
                "p95": _percentile([int(value * 1_000_000) for value in samples], 0.95),
                "p99": _percentile([int(value * 1_000_000) for value in samples], 0.99),
                "maximum": max(samples) if samples else None,
            }
        source_summary = {}
        for key, counts in sorted(self._source_counts.items()):
            first = self._source_first_timestamp.get(key)
            last = self._source_last_timestamp.get(key)
            intervals = self._source_interval_samples.get(key, [])
            elapsed_ms = (last - first) if first is not None and last is not None else 0
            source_summary[f"{key[0]}:{key[1]}"] = {
                **counts,
                "first_source_timestamp_ms": first,
                "last_source_timestamp_ms": last,
                "effective_hz": (
                    counts["fresh"] / (elapsed_ms / 1000.0)
                    if elapsed_ms > 0
                    else None
                ),
                "interval_p50_ms": _plain_percentile(intervals, 0.50),
                "interval_p95_ms": _plain_percentile(intervals, 0.95),
                "interval_p99_ms": _plain_percentile(intervals, 0.99),
                "interval_maximum_ms": max(intervals) if intervals else None,
            }
        return {
            "profiler": "brotato_runtime",
            "enabled": self.enabled,
            "elapsed_seconds": elapsed,
            "stages": stages,
            "counters": dict(sorted(self.counters.items())),
            "values": values,
            "source_summary": source_summary,
            "source_samples": list(self.source_samples),
        }

    def write(self, path: Path, *, wall_seconds: float | None = None) -> dict[str, Any]:
        report = self.report(wall_seconds=wall_seconds)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
