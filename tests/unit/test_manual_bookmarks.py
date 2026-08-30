"""Checks for repeated observation-only in-game bookmarks."""

import json
import sqlite3

import pytest

pytest.importorskip("numpy")

from brotato_ai.data.human_demo import HumanDemoWriter, validate_dataset


def _state(timestamp_ms, tick, *, phase="combat", marks=None):
    return {
        "session": "bookmark-test-session",
        "tick": tick,
        "published_at_ms": timestamp_ms,
        "phase": phase,
        "wave": {"number": 8, "time_left": 20.0},
        "arena": {"width": 1920, "height": 1080},
        "player": {
            "position": {"x": 900.0, "y": 540.0},
            "velocity": {"x": 0.0, "y": 0.0},
            "health": 0.0 if phase == "game_over" else 10.0,
            "max_health": 20.0,
            "radius": 28.0,
        },
        "combat": {"move_speed": 300.0, "weapon_range": 170.0, "weapons": [{"id": "stick"}]},
        "enemies": [],
        "projectiles": [],
        "attack_indicators": [],
        "pickups": [],
        "counters": {"materials": 10, "kills": 1},
        "human_action": 0,
        "human_input": {
            "source": "keyboard",
            "raw_available": False,
            "raw_stick": {"x": 0.0, "y": 0.0},
            "processed_stick": {"x": 0.0, "y": 0.0},
            "buttons": {},
        },
        "ui": {"actions": []},
        "build": {"weapons": [{"id": "stick"}]},
        "dead": phase == "game_over",
        "victory": False,
        "manual_marks": marks or [],
    }


def _mark(marker_id, sequence, reason="manual_bookmark"):
    return {
        "marker_id": marker_id,
        "sequence": sequence,
        "marked_at_ms": 99.0 + sequence,
        "key": "F9",
        "reason": reason,
        "state": {"phase": "combat", "tick": sequence},
    }


def test_repeated_marks_and_terminal_screen_marks_are_preserved(tmp_path):
    dataset = tmp_path / "bookmarks.sqlite"
    with HumanDemoWriter(dataset, session_id="bookmark-test") as writer:
        writer.start_episode(phase="combat", source_session_id="bookmark-test-session")
        writer.record_frame(_state(100.0, 1, marks=[_mark("m1", 1), _mark("m2", 2)]), received_ns=1_000_000_000)
        writer.record_frame(_state(200.0, 2, phase="game_over"), received_ns=1_100_000_000)
        writer.finish_episode(outcome="death", end_phase="game_over")
        assert writer.record_manual_marks_for_last_frame(
            _state(250.0, 2, phase="game_over", marks=[_mark("m3", 3, "manual_death_bookmark")])
        ) == 1
        writer.finalize()

    report = validate_dataset(dataset)
    assert report["manual_marks"] == 3
    connection = sqlite3.connect(str(dataset))
    values = [json.loads(row[0]) for row in connection.execute("SELECT value FROM labels ORDER BY rowid")]
    connection.close()
    assert [value["marker_id"] for value in values] == ["m1", "m2", "m3"]
    assert values[-1]["reason"] == "manual_death_bookmark"
