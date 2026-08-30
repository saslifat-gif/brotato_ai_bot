"""Recorder-only checks for the rich manual-demonstration capture path."""

import sqlite3

import pytest

pytest.importorskip("numpy")

from brotato_ai.data.human_demo import HumanDemoWriter, _from_blob, validate_dataset


def _state(timestamp_ms, tick, *, phase="combat", action=0, health=10.0):
    return {
        "session": "capture-test-session",
        "tick": tick,
        "published_at_ms": timestamp_ms,
        "phase": phase,
        "wave": {"number": 8, "time_left": 20.0},
        "arena": {"width": 1920, "height": 1080},
        "player": {
            "position": {"x": 900.0, "y": 540.0},
            "velocity": {"x": 0.0, "y": 0.0},
            "health": health,
            "max_health": 20.0,
            "radius": 28.0,
        },
        "combat": {
            "move_speed": 300.0,
            "weapon_range": 170.0,
            "weapons": [{"id": "stick"}],
        },
        "enemies": [],
        "projectiles": [],
        "attack_indicators": [],
        "pickups": [],
        "counters": {"materials": 10, "kills": 1},
        "human_action": action,
        "human_input": {
            "source": "keyboard_processed_with_keys",
            "raw_available": False,
            "raw_stick": {"x": 0.0, "y": 0.0},
            "processed_stick": {"x": 0.0, "y": 0.0},
            "buttons": {"key_w": False, "key_a": False, "key_s": False, "key_d": False},
        },
        "ui": {"actions": []},
        "build": {"weapons": [{"id": "stick"}]},
        "dead": phase == "game_over",
        "victory": phase == "victory",
    }


def test_capture_records_reward_boundaries_choices_and_full_hold_duration(tmp_path):
    dataset = tmp_path / "manual.sqlite"
    with HumanDemoWriter(dataset, session_id="capture-test") as writer:
        writer.start_episode(phase="shop", source_session_id="capture-test-session")
        shop = _state(900.0, 0, phase="shop")
        shop["ui"] = {
            "actions": [
                {"id": "item-stick", "role": "buy", "enabled": True, "choice": {"id": "stick"}},
                {"id": "item-spear", "role": "buy", "enabled": True, "choice": {"id": "spear"}},
            ],
        }
        shop["selected_ui_action"] = "item-stick"
        writer.record_frame(shop, received_ns=900_000_000)
        writer.record_raw_sample({**shop, "type": "raw_state"})
        writer.record_frame(_state(1_000.0, 1, action=0), received_ns=1_000_000_000)
        writer.record_frame(_state(1_200.0, 2, action=4), received_ns=1_200_000_000)
        writer.record_frame(_state(1_500.0, 3, action=4), received_ns=1_500_000_000)
        writer.record_frame(_state(1_600.0, 4, phase="game_over", action=4, health=0.0), received_ns=1_600_000_000)
        writer.finish_episode(outcome="death", end_phase="game_over")
        writer.finalize()

    report = validate_dataset(dataset, require_capture=True)
    assert report["ok"] is True
    assert report["capture_ready"] is True
    assert report["stream_status"]["reward_components_present"] is True
    assert report["stream_status"]["fixed_horizon_outcomes_present"] is True
    assert report["stream_status"]["episode_boundaries_closed"] is True

    connection = sqlite3.connect(str(dataset))
    segments = connection.execute(
        "SELECT duration_ms FROM action_segments ORDER BY started_ns"
    ).fetchall()
    reward_blob, derived_blob = connection.execute(
        "SELECT reward_blob,derived_blob FROM frames ORDER BY frame_id LIMIT 1"
    ).fetchone()
    build_available = connection.execute(
        "SELECT available_blob FROM build_decisions LIMIT 1"
    ).fetchone()[0]
    connection.close()
    assert [round(row[0], 1) for row in segments] == [300.0, 400.0]
    assert _from_blob(reward_blob, {}).get("source") == "ApiRewardEngine"
    assert "reward_components" in _from_blob(derived_blob, {})
    assert len(_from_blob(build_available, [])) == 2


def test_capture_preserves_equal_clock_ticks_and_terminal_raw_tail(tmp_path):
    dataset = tmp_path / "terminal.sqlite"
    with HumanDemoWriter(dataset, session_id="capture-terminal") as writer:
        writer.start_episode(phase="combat", source_session_id="capture-test-session")
        writer.record_frame(_state(100.0, 1), received_ns=1_000_000_000)
        # Windows clock resolution can expose equal local timestamps. They
        # remain ordered by frame_id and must not be reported as corruption.
        terminal = _state(200.0, 2, phase="game_over", health=0.0)
        writer.record_frame(terminal, received_ns=1_000_000_000)
        writer.finish_episode(outcome="death", end_phase="game_over")
        writer.record_raw_sample({**terminal, "phase": "EndRun", "type": "raw_state"})
        writer.finalize()

    report = validate_dataset(dataset, require_capture=True)
    assert report["ok"] is True
    assert report["capture_ready"] is True
    connection = sqlite3.connect(str(dataset))
    raw_episode, = connection.execute(
        "SELECT episode_id FROM raw_samples ORDER BY sample_id DESC LIMIT 1"
    ).fetchone()
    connection.close()
    assert raw_episode is not None
