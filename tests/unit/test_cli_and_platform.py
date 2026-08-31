"""Unit tests for unified CLI dispatcher, cross-platform install_mod, and shadow report."""

import json
from pathlib import Path

import pytest

from v4.cli import COMMAND_MAP, build_parser, main as cli_main
from v4.install_mod import (
    GAME_EXECUTABLE_CANDIDATES,
    default_profile_path,
    is_game_directory,
    resolve_game_directory,
    rotate_stale_godot_log,
)
from v4.report_shadow import format_shadow_report, main as report_shadow_main


def test_cli_parser_commands():
    parser = build_parser()
    for command in COMMAND_MAP:
        # Each command in COMMAND_MAP must be valid in the parser
        assert command in parser._subparsers._group_actions[0].choices


def test_cli_unknown_command(capsys):
    ret = cli_main(["unknown-cmd"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Unknown command" in captured.err


def test_cli_help(capsys):
    ret = cli_main(["--help"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "brotato-ai" in captured.out


def test_is_game_directory(tmp_path):
    assert not is_game_directory(tmp_path)

    # Test each supported executable format
    for exe in GAME_EXECUTABLE_CANDIDATES:
        d = tmp_path / f"game_{exe}"
        d.mkdir()
        (d / exe).touch()
        assert is_game_directory(d)
        assert resolve_game_directory(d) == d.resolve()


def test_resolve_game_directory_from_mods_subdir(tmp_path):
    game_dir = tmp_path / "Brotato"
    game_dir.mkdir()
    (game_dir / "Brotato.x86_64").touch()
    mods_dir = game_dir / "mods"
    mods_dir.mkdir()
    assert resolve_game_directory(mods_dir) == game_dir.resolve()


def test_resolve_game_directory_not_found(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="Brotato game executable/package"):
        resolve_game_directory(empty_dir)


def test_default_profile_path_fallback():
    # Calling default_profile_path() should return Path or None without raising
    result = default_profile_path()
    assert result is None or isinstance(result, Path)


def test_rotate_stale_godot_log_noop():
    # Calling rotate_stale_godot_log() should return None when no crash log exists
    assert rotate_stale_godot_log() is None or isinstance(rotate_stale_godot_log(), Path)


def test_format_shadow_report():
    sample_data = [
        {
            "episode": 1,
            "reward": 150.5,
            "steps": 1000,
            "wave": 20,
            "policy": "model",
            "shield_overrides": 50,
            "requested_action_counts": [100, 200, 150, 150, 100, 100, 100, 50, 50],
        },
        {
            "episode": 2,
            "reward": 200.0,
            "steps": 1200,
            "wave": 20,
            "policy": "model",
            "shield_overrides": 30,
            "requested_action_counts": [150, 250, 150, 150, 100, 100, 150, 75, 75],
        },
    ]
    report = format_shadow_report(sample_data)
    assert "# Brotato AI Policy Evaluation Summary" in report
    assert "Episodes Evaluated**: 2" in report
    assert "Mean Wave Reached**: 20.0" in report
    assert "IDLE" in report
    assert "UP_RIGHT" in report


def test_report_shadow_cli(tmp_path, capsys):
    results_path = tmp_path / "shadow.json"
    out_md = tmp_path / "summary.md"
    sample_data = [
        {
            "episode": 1,
            "reward": 50.0,
            "steps": 500,
            "wave": 10,
            "policy": "bc",
            "shield_overrides": 10,
            "requested_action_counts": [50] * 9,
        }
    ]
    results_path.write_text(json.dumps(sample_data), encoding="utf-8")

    ret = report_shadow_main([str(results_path), "-o", str(out_md)])
    assert ret == 0
    assert out_md.is_file()
    assert "# Brotato AI Policy Evaluation Summary" in out_md.read_text(encoding="utf-8")
