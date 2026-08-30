"""Prepare a transition-focused event-model dataset without touching sources.

The rich SQLite recorder is the authoritative source for state, safety, and
fixed-horizon outcomes.  Older semantic JSONL streams contain the same 832
feature contract and genuine human action labels but do not contain rich
state/outcome blobs, so they are imported with explicit provenance and only
the metadata they actually provide.  The output is a new SQLite file.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import uuid
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any


FRAME_COLUMNS = (
    "frame_id", "episode_id", "frame_number", "timestamp_ns", "wall_time_ns",
    "bridge_timestamp_ms", "tick", "phase", "wave", "action", "previous_action",
    "action_segment_id", "state_blob", "input_blob", "controller_blob",
    "derived_blob", "feature_blob", "outcome_blob",
)


def pack(value: Any) -> bytes:
    return zlib.compress(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        level=3,
    )


def unpack(value: bytes | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(zlib.decompress(value).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
        return default


def create_empty_dataset(path: Path) -> None:
    from brotato_ai.data.human_demo import HumanDemoWriter

    HumanDemoWriter(path, session_id="event-training-v2").close()


def copy_source(source: Path, output: Path) -> int:
    source_db = sqlite3.connect(str(source))
    output_db = sqlite3.connect(str(output))
    source_rows = source_db.execute(
        "SELECT " + ",".join(FRAME_COLUMNS) + " FROM frames ORDER BY episode_id,frame_number,frame_id"
    ).fetchall()
    source_episodes = source_db.execute(
        "SELECT episode_id,session_id,started_ns,ended_ns,outcome,start_phase,end_phase FROM episodes"
    ).fetchall()
    output_db.executemany(
        "INSERT OR REPLACE INTO episodes(episode_id,session_id,started_ns,ended_ns,outcome,start_phase,end_phase) VALUES(?,?,?,?,?,?,?)",
        source_episodes,
    )
    last_timestamp: dict[str, int] = {}
    sanitized = []
    for row in source_rows:
        values = list(row)
        episode = str(values[1])
        timestamp = int(values[3])
        previous = last_timestamp.get(episode)
        if previous is not None and timestamp <= previous:
            timestamp = previous + 1
            values[3] = timestamp
        last_timestamp[episode] = timestamp
        sanitized.append(tuple(values))
    output_db.executemany(
        "INSERT INTO frames(" + ",".join(FRAME_COLUMNS) + ") VALUES(" + ",".join("?" for _ in FRAME_COLUMNS) + ")",
        sanitized,
    )
    for table in ("raw_samples", "action_segments", "build_decisions", "transitions", "labels"):
        columns = [row[1] for row in source_db.execute(f"PRAGMA table_info({table})")]
        rows = source_db.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall()
        if rows:
            output_db.executemany(
                f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                rows,
            )
    output_db.commit()
    source_db.close()
    output_db.close()
    return len(source_rows)


def load_semantic(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def semantic_state(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = row.get("counts") if isinstance(row.get("counts"), dict) else {}
    weapon_count = max(0, int(counts.get("weapons", 0) or 0))
    enemy_count = max(0, int(counts.get("enemies", 0) or 0))
    indicator_count = max(0, int(counts.get("indicators", 0) or 0))
    state = {
        "phase": "combat",
        "wave": {"number": int(row.get("wave", 0) or 0)},
        "session": str(row.get("session", "")),
        "combat": {
            "weapon_count": weapon_count,
            "weapons": [{"id": f"semantic_weapon_{index}"} for index in range(weapon_count)],
        },
        "player": {"health": 1.0, "max_health": 1.0},
    }
    derived = {
        "source": "human_semantic_combat_v2",
        "enemy_count": enemy_count,
        "projectile_count": 0,
        "telegraph_count": indicator_count,
        "hazard_actionable": bool(indicator_count),
        "semantic_counts": counts,
        "nearest_enemy_distance": None,
    }
    return state, derived


def append_semantic(output: Path, paths: list[Path]) -> tuple[int, int]:
    db = sqlite3.connect(str(output))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in load_semantic(path):
            try:
                features = row.get("features", [])
                action = int(row.get("action", -1))
                previous = int(row.get("previous_action", -1))
                if len(features) != 832 or not (0 <= action < 9 and 0 <= previous < 9):
                    continue
                key = f"semantic:{row.get('session','')}:{int(row.get('episode',0))}"
                row = dict(row)
                row["_source_path"] = str(path)
                row["_episode_key"] = key
                groups[key].append(row)
            except (TypeError, ValueError):
                continue

    frame_count = 0
    transition_count = 0
    for episode_id, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (float(row.get("timestamp", 0.0)), int(row.get("tick", 0))))
        last_timestamp = 0
        last_segment_action = None
        segment_id = None
        segment_start = 0
        segment_rows: list[tuple[str, int, int, int, int | None, float | None]] = []
        for frame_number, row in enumerate(rows):
            timestamp_ns = int(float(row.get("timestamp", 0.0)) * 1_000_000_000)
            timestamp_ns = max(timestamp_ns, last_timestamp + 1)
            last_timestamp = timestamp_ns
            action = int(row["action"])
            previous = int(row["previous_action"])
            if action != previous:
                transition_count += 1
            if segment_id is None or action != last_segment_action:
                if segment_id is not None:
                    segment_rows[-1] = (
                        segment_rows[-1][0], segment_rows[-1][1], segment_rows[-1][2],
                        segment_rows[-1][3], timestamp_ns,
                        (timestamp_ns - segment_rows[-1][2]) / 1_000_000.0,
                    )
                segment_id = uuid.uuid4().hex
                segment_start = timestamp_ns
                segment_rows.append((segment_id, episode_id, segment_start, action, None, None))
                last_segment_action = action
            state, derived = semantic_state(row)
            input_blob = {
                "source": "human_semantic_combat_v2",
                "raw_available": False,
                "human_input_age_ms": row.get("human_input_age_ms"),
                "session": row.get("session"),
            }
            controller_blob = {
                "source": "human_semantic_combat_v2",
                "candidate_risks": {},
                "unsafe_alternative_labels": "not_available_in_semantic_stream",
            }
            features = [round(float(value), 6) for value in row["features"]]
            db.execute(
                "INSERT INTO frames(episode_id,frame_number,timestamp_ns,wall_time_ns,bridge_timestamp_ms,tick,phase,wave,action,previous_action,action_segment_id,state_blob,input_blob,controller_blob,derived_blob,feature_blob,outcome_blob) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    episode_id, frame_number, timestamp_ns, timestamp_ns,
                    timestamp_ns / 1_000_000.0, int(row.get("tick", -1) or -1),
                    "combat", int(row.get("wave", 0) or 0), action, previous,
                    segment_id, pack(state), pack(input_blob), pack(controller_blob),
                    pack(derived), pack(features), None,
                ),
            )
            frame_count += 1
        if rows:
            db.execute(
                "INSERT OR REPLACE INTO episodes(episode_id,session_id,started_ns,ended_ns,outcome,start_phase,end_phase) VALUES(?,?,?,?,?,?,?)",
                (
                    episode_id, str(rows[0].get("session", "")),
                    int(float(rows[0].get("timestamp", 0.0)) * 1_000_000_000),
                    last_timestamp, "imported_human_stream", "combat", "combat",
                ),
            )
        for segment_id, segment_episode, started_ns, segment_action, ended_ns, duration_ms in segment_rows:
            db.execute(
                "INSERT OR REPLACE INTO action_segments(segment_id,episode_id,action,started_ns,ended_ns,duration_ms) VALUES(?,?,?,?,?,?)",
                (segment_id, segment_episode, segment_action, started_ns, ended_ns, duration_ms),
            )
    db.commit()
    db.close()
    return frame_count, transition_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new merged event-model training SQLite dataset")
    parser.add_argument("--rich-dataset", type=Path, required=True)
    parser.add_argument("--semantic-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if not args.rich_dataset.is_file():
        raise SystemExit(f"rich dataset not found: {args.rich_dataset}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_empty_dataset(args.output)
    rich_frames = copy_source(args.rich_dataset, args.output)
    semantic_frames, semantic_transitions = append_semantic(args.output, args.semantic_jsonl)
    db = sqlite3.connect(str(args.output))
    metadata = {
        "dataset": "brotato_event_human_training_v2",
        "source_rich_sqlite": str(args.rich_dataset),
        "source_semantic_jsonl": [str(path) for path in args.semantic_jsonl],
        "source_rich_frames": rich_frames,
        "source_semantic_frames": semantic_frames,
        "source_semantic_transitions": semantic_transitions,
        "timestamp_sanitization": "non-increasing source frame timestamps bumped by 1 ns in the new copy",
        "semantic_stream_limitations": "no rich state, candidate action risks, or fixed-horizon outcomes; provenance retained in blobs",
    }
    db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", ("dataset", json.dumps(metadata, separators=(",", ":"))))
    db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", ("schema_version", json.dumps(1)))
    db.commit()
    db.close()
    print(json.dumps({"output": str(args.output), "rich_frames": rich_frames, "semantic_frames": semantic_frames, "semantic_transitions": semantic_transitions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
