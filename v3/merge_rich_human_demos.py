"""Merge completed rich human-demo SQLite captures for offline training.

The recorder intentionally creates one SQLite file per manual playthrough.
This utility combines those files without rewriting the sources, preserving
raw samples, rich frames, build choices, action segments, transitions, labels,
and episode outcomes.  Primary-key references are remapped because each
capture starts its own AUTOINCREMENT frame/sample sequence.

This is an offline data-preparation tool.  It never connects to the game and
does not change any controller or policy mode.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


EPISODE_COLUMNS = (
    "episode_id", "session_id", "started_ns", "ended_ns", "outcome",
    "start_phase", "end_phase", "first_frame_id", "last_frame_id",
    "first_timestamp_ns", "last_timestamp_ns", "first_bridge_timestamp_ms",
    "last_bridge_timestamp_ms",
)
FRAME_COLUMNS = (
    "episode_id", "frame_number", "timestamp_ns", "wall_time_ns",
    "bridge_timestamp_ms", "tick", "phase", "wave", "action",
    "previous_action", "action_segment_id", "state_blob", "input_blob",
    "controller_blob", "derived_blob", "feature_blob", "outcome_blob",
    "reward_blob",
)
RAW_COLUMNS = (
    "session_id", "episode_id", "bridge_session_id", "timestamp_ns",
    "bridge_timestamp_ms", "tick", "state_blob", "input_blob",
)
SEGMENT_COLUMNS = (
    "segment_id", "episode_id", "action", "started_ns", "ended_ns",
    "duration_ms",
)
BUILD_COLUMNS = (
    "episode_id", "frame_id", "timestamp_ns", "phase", "available_blob",
    "build_before_blob", "build_after_blob", "chosen_action", "source",
)
LABEL_COLUMNS = ("frame_id", "label", "value", "annotator", "created_ns")


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def select_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Iterable[str],
    *,
    order_by: str | None = None,
) -> list[tuple[Any, ...]]:
    available = table_columns(connection, table)
    expressions = [column if column in available else f"NULL AS {column}" for column in columns]
    query = f"SELECT {','.join(expressions)} FROM {table}"
    if order_by:
        query += f" ORDER BY {order_by}"
    return connection.execute(query).fetchall()


def create_output(path: Path) -> None:
    from brotato_ai.data.human_demo import HumanDemoWriter

    writer = HumanDemoWriter(path, session_id="merged-human-demos")
    writer.close()


def prefixed(prefix: str, value: Any) -> str:
    return f"{prefix}:{value}"


def merge_sources(sources: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dataset files not found: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    create_output(output)
    destination = sqlite3.connect(str(output))
    source_summaries: list[dict[str, Any]] = []
    try:
        for source_index, source_path in enumerate(sources, start=1):
            prefix = f"merged{source_index}"
            source = sqlite3.connect(str(source_path))
            try:
                episode_rows = select_rows(
                    source, "episodes", EPISODE_COLUMNS, order_by="started_ns,episode_id"
                )
                episode_map = {
                    str(row[0]): prefixed(prefix, row[0]) for row in episode_rows
                }
                frame_map: dict[int, int] = {}
                segment_map: dict[str, str] = {}

                frame_rows = source.execute(
                    "SELECT frame_id," + ",".join(FRAME_COLUMNS)
                    + " FROM frames ORDER BY episode_id,frame_number,frame_id"
                ).fetchall()
                frame_insert = destination.cursor()
                for row in frame_rows:
                    old_frame_id = int(row[0])
                    values = list(row[1:])
                    values[0] = episode_map.get(str(values[0]), prefixed(prefix, values[0]))
                    if values[10] is not None:
                        old_segment = str(values[10])
                        values[10] = segment_map.setdefault(old_segment, prefixed(prefix, old_segment))
                    frame_insert.execute(
                        "INSERT INTO frames(" + ",".join(FRAME_COLUMNS) + ") VALUES("
                        + ",".join("?" for _ in FRAME_COLUMNS) + ")",
                        tuple(values),
                    )
                    frame_map[old_frame_id] = int(frame_insert.lastrowid)

                for row in episode_rows:
                    old_episode_id = str(row[0])
                    values = list(row)
                    values[0] = episode_map[old_episode_id]
                    if values[7] is not None:
                        values[7] = frame_map.get(int(values[7]))
                    if values[8] is not None:
                        values[8] = frame_map.get(int(values[8]))
                    destination.execute(
                        "INSERT INTO episodes(" + ",".join(EPISODE_COLUMNS) + ") VALUES("
                        + ",".join("?" for _ in EPISODE_COLUMNS) + ")",
                        tuple(values),
                    )

                raw_rows = select_rows(source, "raw_samples", RAW_COLUMNS, order_by="timestamp_ns,sample_id")
                destination.executemany(
                    "INSERT INTO raw_samples(" + ",".join(RAW_COLUMNS) + ") VALUES("
                    + ",".join("?" for _ in RAW_COLUMNS) + ")",
                    [
                        (
                            row[0],
                            episode_map.get(str(row[1])) if row[1] is not None else None,
                            row[2], row[3], row[4], row[5], row[6], row[7],
                        )
                        for row in raw_rows
                    ],
                )

                segment_rows = select_rows(source, "action_segments", SEGMENT_COLUMNS, order_by="started_ns,segment_id")
                destination.executemany(
                    "INSERT INTO action_segments(" + ",".join(SEGMENT_COLUMNS) + ") VALUES("
                    + ",".join("?" for _ in SEGMENT_COLUMNS) + ")",
                    [
                        (
                            segment_map.setdefault(str(row[0]), prefixed(prefix, row[0])),
                            episode_map.get(str(row[1]), prefixed(prefix, row[1])),
                            row[2], row[3], row[4], row[5],
                        )
                        for row in segment_rows
                    ],
                )

                build_rows = select_rows(source, "build_decisions", BUILD_COLUMNS, order_by="timestamp_ns,decision_id")
                destination.executemany(
                    "INSERT INTO build_decisions(" + ",".join(BUILD_COLUMNS) + ") VALUES("
                    + ",".join("?" for _ in BUILD_COLUMNS) + ")",
                    [
                        (
                            episode_map.get(str(row[0]), prefixed(prefix, row[0])),
                            frame_map.get(int(row[1]), int(row[1])) if row[1] is not None else None,
                            row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                        )
                        for row in build_rows
                    ],
                )

                transition_columns = table_columns(source, "transitions")
                if transition_columns:
                    transition_rows = source.execute(
                        "SELECT frame_id,outcome_50ms,outcome_100ms,outcome_250ms,outcome_500ms,outcome_1000ms "
                        "FROM transitions ORDER BY frame_id"
                    ).fetchall()
                    destination.executemany(
                        "INSERT INTO transitions(frame_id,outcome_50ms,outcome_100ms,outcome_250ms,outcome_500ms,outcome_1000ms) "
                        "VALUES(?,?,?,?,?,?)",
                        [
                            (frame_map.get(int(row[0]), int(row[0])), row[1], row[2], row[3], row[4], row[5])
                            for row in transition_rows
                        ],
                    )

                label_rows = select_rows(source, "labels", LABEL_COLUMNS, order_by="created_ns,label_id")
                destination.executemany(
                    "INSERT INTO labels(" + ",".join(LABEL_COLUMNS) + ") VALUES("
                    + ",".join("?" for _ in LABEL_COLUMNS) + ")",
                    [
                        (frame_map.get(int(row[0]), int(row[0])), row[1], row[2], row[3], row[4])
                        for row in label_rows
                    ],
                )

                source_summaries.append({
                    "path": str(source_path),
                    "episodes": len(episode_rows),
                    "frames": len(frame_rows),
                    "raw_samples": len(raw_rows),
                    "action_segments": len(segment_rows),
                    "build_decisions": len(build_rows),
                })
                destination.commit()
            finally:
                source.close()

        metadata = {
            "dataset": "brotato_event_human_training_merged",
            "sources": source_summaries,
            "source_count": len(source_summaries),
            "id_remapping": "episode, frame, action-segment, build-decision, transition, and label references remapped; source files unchanged",
            "controller_policy_mode": "HANDCRAFTED during capture; no policy control was enabled",
        }
        destination.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            ("merged_sources", json.dumps(metadata, separators=(",", ":"))),
        )
        destination.commit()
    finally:
        destination.close()
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge rich human-demo SQLite captures for offline training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    result = merge_sources(args.datasets, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
