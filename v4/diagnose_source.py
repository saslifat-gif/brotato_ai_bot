"""Measure fresh game-state delivery without running the controller.

The active bridge is a push-based TCP stream.  This diagnostic therefore does
not invent polling requests: it measures socket receive cadence, source/game
timestamps, framing, JSON parsing, and entity complexity separately.  With no
recording argument it connects to the mod's independent raw-state port and
acts as a minimal reader.  With a JSONL argument it analyzes an existing raw
recording using the source timestamps preserved in each row.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECTILE_BUCKETS = (
    ("0-10", 0, 10),
    ("11-25", 11, 25),
    ("26-50", 26, 50),
    ("51-100", 51, 100),
    ("100+", 101, math.inf),
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _stats(values: Iterable[float], *, unit: str = "ms") -> dict[str, Any]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "maximum": None, "unit": unit}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values),
        "unit": unit,
    }


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _source_timestamp(message: Mapping[str, Any]) -> int | None:
    for key in ("published_at_ms", "game_timestamp_ms", "timestamp_ms"):
        value = _optional_int(message.get(key))
        if value is not None and value >= 0:
            return value
    return None


def _source_sequence(message: Mapping[str, Any]) -> int | None:
    for key in ("source_sequence", "raw_sequence"):
        value = _optional_int(message.get(key))
        if value is not None:
            return value
    return None


def _source_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    session = str(message.get("session", ""))
    sequence = _source_sequence(message)
    if sequence is not None:
        return (session, "sequence", sequence)
    timestamp = _source_timestamp(message)
    if timestamp is not None:
        return (session, "timestamp", timestamp)
    tick = _optional_int(message.get("tick"))
    if tick is not None:
        return (session, "tick", tick)
    return (session, "payload", json.dumps(dict(message), sort_keys=True, separators=(",", ":")))


def _entity_count(message: Mapping[str, Any], key: str) -> int:
    value = message.get(key)
    return len(value) if isinstance(value, (list, tuple)) else 0


def _bucket_name(projectiles: int) -> str:
    for name, lower, upper in PROJECTILE_BUCKETS:
        if lower <= projectiles <= upper:
            return name
    return "100+"


class SourceCollector:
    """Accumulate fresh-state and transport measurements with bounded samples."""

    def __init__(self, *, sample_limit: int = 20_000):
        self.sample_limit = max(100, int(sample_limit))
        self.samples: list[dict[str, Any]] = []
        self.source_intervals_ms: list[float] = []
        self.receive_intervals_ms: list[float] = []
        self.response_latencies_ms: list[float] = []
        self.parse_times_ms: list[float] = []
        self.payload_sizes: list[float] = []
        self.fresh_count = 0
        self.duplicate_count = 0
        self.total_count = 0
        self._previous_key: tuple[Any, ...] | None = None
        self._previous_source_timestamp: int | None = None
        self._previous_receive_ns: int | None = None
        self._bucket_intervals: dict[str, list[float]] = {
            name: [] for name, _, _ in PROJECTILE_BUCKETS
        }
        self._bucket_latencies: dict[str, list[float]] = {
            name: [] for name, _, _ in PROJECTILE_BUCKETS
        }
        self._bucket_counts: dict[str, int] = {
            name: 0 for name, _, _ in PROJECTILE_BUCKETS
        }

    def add(
        self,
        message: Mapping[str, Any],
        *,
        payload_size_bytes: int | None = None,
        local_receive_ns: int | None = None,
        response_latency_ms: float | None = None,
        parse_ms: float | None = None,
        payload_complete_ns: int | None = None,
        receive_call_start_ns: int | None = None,
        recv_start_ns: int | None = None,
        parse_start_ns: int | None = None,
        parse_end_ns: int | None = None,
    ) -> None:
        if message.get("type") not in {"raw_state", "state"}:
            return
        self.total_count += 1
        timestamp = _source_timestamp(message)
        source_sequence = _source_sequence(message)
        key = _source_key(message)
        fresh = self._previous_key is None or key != self._previous_key
        if fresh:
            self.fresh_count += 1
            if self._previous_source_timestamp is not None and timestamp is not None:
                interval = timestamp - self._previous_source_timestamp
                if interval > 0:
                    interval_float = float(interval)
                    self.source_intervals_ms.append(interval_float)
                    bucket = _bucket_name(_entity_count(message, "projectiles"))
                    self._bucket_intervals[bucket].append(interval_float)
            if (
                self._previous_receive_ns is not None
                and local_receive_ns is not None
                and local_receive_ns > self._previous_receive_ns
            ):
                self.receive_intervals_ms.append(
                    (local_receive_ns - self._previous_receive_ns) / 1_000_000.0
                )
        else:
            self.duplicate_count += 1
        if fresh:
            if timestamp is not None:
                self._previous_source_timestamp = timestamp
            if local_receive_ns is not None:
                self._previous_receive_ns = local_receive_ns
            self._previous_key = key

        projectile_count = _entity_count(message, "projectiles")
        self._bucket_counts[_bucket_name(projectile_count)] += 1
        payload_size = payload_size_bytes
        if payload_size is None:
            payload_size = len(
                json.dumps(dict(message), separators=(",", ":"), allow_nan=False).encode("utf-8")
            )
        self.payload_sizes.append(float(payload_size))
        if response_latency_ms is not None and math.isfinite(float(response_latency_ms)):
            self.response_latencies_ms.append(float(response_latency_ms))
            self._bucket_latencies[_bucket_name(projectile_count)].append(float(response_latency_ms))
        if parse_ms is not None and math.isfinite(float(parse_ms)):
            self.parse_times_ms.append(float(parse_ms))

        if len(self.samples) < self.sample_limit:
            sample = {
                "message_type": str(message.get("type", "")),
                "session": str(message.get("session", "")),
                "source_timestamp_ms": timestamp,
                "source_sequence": source_sequence,
                "source_tick": _optional_int(message.get("tick")),
                "fresh": fresh,
                "local_receive_time_ns": local_receive_ns,
                "payload_complete_ns": payload_complete_ns,
                "receive_call_start_ns": receive_call_start_ns,
                "recv_start_ns": recv_start_ns,
                "parse_start_ns": parse_start_ns,
                "parse_end_ns": parse_end_ns,
                "response_latency_ms": response_latency_ms,
                "parse_ms": parse_ms,
                "payload_size_bytes": payload_size,
                "enemy_count": _entity_count(message, "enemies"),
                "projectile_count": projectile_count,
            }
            self.samples.append(sample)

    def report(self, *, transport: str, elapsed_wall_seconds: float | None = None) -> dict[str, Any]:
        source_stats = _stats(self.source_intervals_ms)
        if source_stats["mean"]:
            source_stats["effective_hz"] = 1000.0 / source_stats["mean"]
        else:
            source_stats["effective_hz"] = None
        receive_stats = _stats(self.receive_intervals_ms)
        if receive_stats["mean"]:
            receive_stats["effective_hz"] = 1000.0 / receive_stats["mean"]
        else:
            receive_stats["effective_hz"] = None
        buckets = {}
        for name, _, _ in PROJECTILE_BUCKETS:
            intervals = self._bucket_intervals[name]
            latencies = self._bucket_latencies[name]
            bucket_source = _stats(intervals)
            bucket_source["effective_hz"] = 1000.0 / bucket_source["mean"] if bucket_source["mean"] else None
            buckets[name] = {
                "states": self._bucket_counts[name],
                "fresh_intervals_ms": bucket_source,
                "response_latency_ms": _stats(latencies),
            }
        return {
            "diagnostic": "source_boundary",
            "transport": {
                "model": transport,
                "is_poll_based": False,
                "requests": None,
                "requests_per_second": None,
                "note": "The active bridge is a push TCP stream; there is no client request to poll faster.",
            },
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "records": {
                "received": self.total_count,
                "fresh": self.fresh_count,
                "duplicates": self.duplicate_count,
                "fresh_fraction": self.fresh_count / self.total_count if self.total_count else None,
                "fresh_states_per_wall_second": (
                    self.fresh_count / elapsed_wall_seconds
                    if elapsed_wall_seconds and elapsed_wall_seconds > 0
                    else None
                ),
            },
            "fresh_state_intervals_ms": source_stats,
            "local_receive_intervals_ms": receive_stats,
            "response_latency_ms": _stats(self.response_latencies_ms),
            "parse_ms": _stats(self.parse_times_ms),
            "payload_size_bytes": _stats(self.payload_sizes, unit="bytes"),
            "projectile_buckets": buckets,
            "samples": self.samples,
        }


def analyze_recording(path: Path, *, sample_limit: int) -> dict[str, Any]:
    collector = SourceCollector(sample_limit=sample_limit)
    started = time.perf_counter()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                message = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(message, dict):
                continue
            recorded_at_ms = _optional_int(message.get("recorded_at_ms"))
            collector.add(
                message,
                payload_size_bytes=len(line.encode("utf-8")),
                local_receive_ns=recorded_at_ms * 1_000_000 if recorded_at_ms is not None else None,
            )
    report = collector.report(
        transport="push_tcp_recording",
        elapsed_wall_seconds=time.perf_counter() - started,
    )
    report["input"] = str(path)
    report["interpretation"] = {
        "source_clock": "published_at_ms from the game/mod",
        "local_clock": "recorded_at_ms when present; this is the recorder-side arrival proxy",
        "warning": "Offline file-read wall time is not a transport rate; use source and recorded intervals.",
    }
    return report


def read_live_stream(
    *,
    host: str,
    port: int,
    seconds: float,
    max_records: int,
    sample_limit: int,
) -> dict[str, Any]:
    collector = SourceCollector(sample_limit=sample_limit)
    started = time.perf_counter()
    deadline = started + max(0.1, float(seconds)) if seconds > 0 else None
    buffer = bytearray()
    buffer_received_ns: int | None = None
    buffer_recv_start_ns: int | None = None
    with socket.create_connection((host, int(port)), timeout=30.0) as connection:
        connection.settimeout(0.25)
        while (max_records <= 0 or collector.total_count < max_records) and (
            deadline is None or time.perf_counter() < deadline
        ):
            recv_start_ns = time.perf_counter_ns()
            try:
                chunk = connection.recv(1024 * 1024)
            except socket.timeout:
                continue
            if not chunk:
                break
            response_arrival_ns = time.perf_counter_ns()
            if not buffer:
                buffer_received_ns = response_arrival_ns
                buffer_recv_start_ns = recv_start_ns
            buffer.extend(chunk)
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                payload_complete_ns = buffer_received_ns or response_arrival_ns
                line_recv_start_ns = buffer_recv_start_ns
                if not buffer:
                    buffer_received_ns = None
                    buffer_recv_start_ns = None
                if not line.strip():
                    continue
                parse_start_ns = time.perf_counter_ns()
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                parse_end_ns = time.perf_counter_ns()
                if isinstance(message, dict):
                    collector.add(
                        message,
                        payload_size_bytes=len(line),
                        local_receive_ns=response_arrival_ns,
                        response_latency_ms=(
                            (response_arrival_ns - line_recv_start_ns) / 1_000_000.0
                            if line_recv_start_ns is not None
                            else None
                        ),
                        parse_ms=(parse_end_ns - parse_start_ns) / 1_000_000.0,
                        payload_complete_ns=payload_complete_ns,
                        receive_call_start_ns=recv_start_ns,
                        recv_start_ns=line_recv_start_ns,
                        parse_start_ns=parse_start_ns,
                        parse_end_ns=parse_end_ns,
                    )
                if max_records > 0 and collector.total_count >= max_records:
                    break
    elapsed = time.perf_counter() - started
    report = collector.report(transport="push_tcp_live_minimal_reader", elapsed_wall_seconds=elapsed)
    report["input"] = {"host": host, "port": int(port), "seconds_requested": seconds, "max_records": max_records}
    report["interpretation"] = {
        "source_clock": "published_at_ms from the game/mod",
        "local_clock": "perf_counter_ns immediately after socket.recv returned",
        "minimal_reader": "No controller, feature computation, reward, action, or disk recording was run.",
        "polling_test": "Not applicable because this transport pushes states over a persistent TCP stream.",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", type=Path, help="existing raw JSONL recording; omit for a live raw-port reader")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4243)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.recording is not None:
        report = analyze_recording(args.recording, sample_limit=args.sample_limit)
    else:
        report = read_live_stream(
            host=args.host,
            port=args.port,
            seconds=args.seconds,
            max_records=max(0, args.max_records),
            sample_limit=args.sample_limit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "received": report["records"]["received"],
        "fresh": report["records"]["fresh"],
        "source_mean_ms": report["fresh_state_intervals_ms"]["mean"],
        "source_effective_hz": report["fresh_state_intervals_ms"].get("effective_hz"),
        "receive_mean_ms": report["local_receive_intervals_ms"]["mean"],
        "receive_effective_hz": report["local_receive_intervals_ms"].get("effective_hz"),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
