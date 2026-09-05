import pytest

from brotato_ai.evaluation.full_run import FullRunMetrics, summarize, validate_start
from v4.ui_automation import available_actions


def test_ban_button_is_never_an_affordable_purchase():
    state = {"ui": {"actions": [
        {"id": "/root/Shop/BuyButton", "role": "buy", "name": "BuyButton", "enabled": True},
        {"id": "/root/Shop/BanButton", "role": "buy", "name": "BanButton", "enabled": True},
    ]}}
    assert [a["name"] for a in available_actions(state, "buy")] == ["BuyButton"]


def test_only_genuine_terminal_outcomes_enter_win_rate():
    metrics = FullRunMetrics()
    win = metrics.finish({"victory": True}, terminated=True, truncated=False, elapsed=10)
    death = metrics.finish({"dead": True}, terminated=True, truncated=False, elapsed=10)
    interrupted = metrics.finish({}, terminated=False, truncated=True, elapsed=10)
    inconsistent = metrics.finish({"dead": True, "victory": True}, terminated=True, truncated=False, elapsed=10)
    result = summarize([win, death, interrupted, inconsistent])
    assert result["valid_runs"] == 2 and result["incomplete_runs"] == 2
    assert result["wins"] == 1 and result["win_rate"] == .5
    assert summarize([interrupted])["win_rate"] is None


@pytest.mark.parametrize("change", [
    {"phase": "shop"}, {"wave": {"number": 8}},
    {"wave": {"number": 1, "duration": 20, "time_left": 10}},
    {"build": {"character_id": "engineer", "weapons": [{"id": "weapon_smg"}]}},
    {"build": {"character_id": "character_well_rounded", "weapons": [{"id": "weapon_wrench"}]}},
])
def test_full_run_rejects_mismatched_or_partial_start(change):
    state = {"phase": "combat", "wave": {"number": 1}, "build": {
        "character_id": "character_well_rounded", "weapons": [{"id": "weapon_smg"}]}}
    state.update(change)
    with pytest.raises(ValueError):
        validate_start(state, character="character_well_rounded", weapon="weapon_smg")


def test_valid_start_records_unknown_difficulty_without_inventing_it():
    state = {"phase": "combat", "wave": {"number": 1}, "build": {
        "character_id": "character_well_rounded", "weapons": [{"id": "weapon_smg_1"}]}}
    assert validate_start(state, character="character_well_rounded", weapon="weapon_smg")["difficulty"] is None
