"""Profile the structured runtime against representative recorded states.

This benchmark measures processing, not network delivery.  The report keeps
the source timestamp interval separate from local processing timings so a
slow producer cannot be mistaken for a slow controller.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from brotato_ai.control import CombatDecisionPipeline, CombatSafetyShield, CrowdRecoveryGuard
from brotato_ai.data.replay import JsonlReplay
from brotato_ai.data.recorder import DecisionTraceLogger
from brotato_ai.domain.state import StateSnapshot
from brotato_ai.performance import RuntimeProfiler
from v4.combat_policy import HierarchicalCombatVectorizer


def _raw_lines(path: Path, max_records: int) -> list[str]:
    lines: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(raw, dict) and raw.get("type") in {"raw_state", "state"}:
                lines.append(line)
                if max_records > 0 and len(lines) >= max_records:
                    break
    return lines


def _counts(raw_line: str) -> tuple[int, int]:
    raw = json.loads(raw_line)
    return len(raw.get("enemies", [])), len(raw.get("projectiles", []))


def _select_cases(lines: list[str]) -> dict[str, list[str]]:
    if not lines:
        return {}
    enemy_counts = [_counts(line)[0] for line in lines]
    projectile_counts = [_counts(line)[1] for line in lines]
    enemy_cut = sorted(enemy_counts)[max(0, int(len(enemy_counts) * 0.90) - 1)]
    projectile_cut = sorted(projectile_counts)[max(0, int(len(projectile_counts) * 0.90) - 1)]
    dense_enemy = [line for line in lines if _counts(line)[0] >= enemy_cut]
    dense_projectile = [line for line in lines if _counts(line)[1] >= projectile_cut]
    return {
        "normal": lines,
        "dense_enemy": dense_enemy or lines,
        "dense_projectile": dense_projectile or lines,
    }


def _run_case(
    lines: Iterable[str],
    *,
    recording: bool,
    sample_limit: int,
    source_intervals_are_contiguous: bool,
) -> dict[str, Any]:
    profiler = RuntimeProfiler(enabled=True, sample_limit=sample_limit)
    shield = CombatSafetyShield(enabled=True)
    pipeline = CombatDecisionPipeline(
        safety_shield=shield,
        crowd_recovery_guard=CrowdRecoveryGuard(enabled=True, shield=shield),
    )
    vectorizer = HierarchicalCombatVectorizer()
    previous_action = 0
    logger = None
    temporary = None
    if recording:
        temporary = tempfile.NamedTemporaryFile(prefix="brotato-runtime-", suffix=".jsonl", delete=False)
        temporary.close()
        logger = DecisionTraceLogger(Path(temporary.name), profiler=profiler)
    wall_start = time.perf_counter_ns()
    count = 0
    previous_published_ms: int | None = None
    try:
        for line in lines:
            loop_started = profiler.begin("control_loop_total")
            started = profiler.begin("json_decode")
            raw = json.loads(line)
            profiler.end("json_decode", started)
            started = profiler.begin("state_normalization")
            snapshot = StateSnapshot.from_payload(raw)
            profiler.end("state_normalization", started)
            profiler.count("source_states")
            if previous_published_ms is not None and snapshot.timestamp_ms > previous_published_ms:
                interval_name = (
                    "source_interval_ms"
                    if source_intervals_are_contiguous
                    else "selected_row_interval_ms"
                )
                profiler.value(interval_name, snapshot.timestamp_ms - previous_published_ms)
            previous_published_ms = snapshot.timestamp_ms
            profiler.count("enemy_entities", len(snapshot.enemies))
            profiler.count("projectile_entities", len(snapshot.projectiles))
            started = profiler.begin("decision_pipeline")
            trace = pipeline.apply(snapshot, int(raw.get("action", previous_action)), previous_action=previous_action)
            profiler.end("decision_pipeline", started)
            previous_action = trace.decision.applied_action
            started = profiler.begin("observation_vectorizer")
            vectorizer.build(snapshot.payload, previous_action)
            profiler.end("observation_vectorizer", started)
            if logger is not None:
                started = profiler.begin("recording_trace_serialization_and_io")
                logger.record(trace)
                profiler.end("recording_trace_serialization_and_io", started)
            count += 1
            profiler.end("control_loop_total", loop_started)
    finally:
        if logger is not None:
            logger.close()
        if temporary is not None:
            Path(temporary.name).unlink(missing_ok=True)
    total_start = wall_start
    total_elapsed = max(1e-9, (time.perf_counter_ns() - total_start) / 1_000_000_000.0)
    report = profiler.report(wall_seconds=total_elapsed)
    report.update({
        "records": count,
        "recording_enabled": recording,
        "source_rate_hz": (
            1000.0 / report["values"]["source_interval_ms"]["mean"]
            if report["values"].get("source_interval_ms", {}).get("mean")
            else None
        ),
        "source_rate_scope": (
            "contiguous_recording_rows"
            if source_intervals_are_contiguous
            else "noncontiguous_complexity_subset; not a source-rate measurement"
        ),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the structured Brotato runtime")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--sample-limit", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = _raw_lines(args.recording, max(0, args.max_records) + max(0, args.warmup))
    if args.warmup and len(lines) > args.warmup:
        lines = lines[args.warmup:]
    if not lines:
        raise SystemExit("recording contains no state rows")
    cases = _select_cases(lines)
    report: dict[str, Any] = {
        "recording": str(args.recording.resolve()),
        "records": len(lines),
        "cases": {},
    }
    for name, case_lines in cases.items():
        report["cases"][name] = {
            "recording_off": _run_case(
                case_lines,
                recording=False,
                sample_limit=args.sample_limit,
                source_intervals_are_contiguous=name == "normal",
            ),
            "recording_on": _run_case(
                case_lines,
                recording=True,
                sample_limit=args.sample_limit,
                source_intervals_are_contiguous=name == "normal",
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
