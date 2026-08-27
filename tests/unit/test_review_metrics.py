from types import SimpleNamespace

import numpy as np
import pytest

from brotato_ai.control.hazards import UnifiedHazardScorer
from brotato_ai.domain.actions import MoveAction
from brotato_ai.evaluation.temporal_ablation import ablation_observations
from brotato_ai.telemetry import percentile, risk_diagnostics, reward_time_scale


def test_percentile_is_deterministic_and_empty_safe():
    assert percentile([], 0.95) == 0.0
    assert percentile([10.0, 20.0, 30.0], 0.95) == pytest.approx(29.0)
    assert percentile([float("nan"), 7.0], 0.5) == 7.0


def test_reward_time_scale_uses_reference_time_and_bounds():
    assert reward_time_scale(1000.0 / 24.0, 24.0) == pytest.approx(1.0)
    assert reward_time_scale(1.0, 24.0) == pytest.approx(0.25)
    assert reward_time_scale(1000.0, 24.0) == pytest.approx(4.0)


def test_risk_diagnostics_reports_regret_and_unsafe_fraction():
    risks = {0: SimpleNamespace(total=0.1), 1: SimpleNamespace(total=0.8), 2: SimpleNamespace(total=0.7)}
    result = risk_diagnostics(risks, 1)
    assert result["minimum_action_risk"] == pytest.approx(0.1)
    assert result["unsafe_action_count"] == 2
    assert result["unsafe_action_fraction"] == pytest.approx(2 / 3)
    assert result["requested_to_minimum_regret"] == pytest.approx(0.7)


def test_history_ablations_only_change_the_declared_slice():
    observations = np.arange(48, dtype=np.float32).reshape(3, 16)
    variants = ablation_observations(observations, history_start=4, history_size=4, seed=17)
    np.testing.assert_array_equal(variants["base"], observations)
    np.testing.assert_array_equal(variants["history_zeroed"][:, :4], observations[:, :4])
    np.testing.assert_array_equal(variants["history_zeroed"][:, 8:], observations[:, 8:])
    assert np.all(variants["history_zeroed"][:, 4:8] == 0.0)
    assert variants["history_shuffled"].shape == observations.shape

def test_human_anchor_coefficient_anneals_linearly():
    from v3.train_combat_finetune import HumanAnchoredPPO

    model = object.__new__(HumanAnchoredPPO)
    model.bc_coefficient = 0.20
    model.bc_final_coefficient = 0.0
    model.bc_anneal_steps = 10_000
    model.num_timesteps = 0
    assert model._effective_bc_coefficient() == pytest.approx(0.20)
    model.num_timesteps = 5_000
    assert model._effective_bc_coefficient() == pytest.approx(0.10)
    model.num_timesteps = 10_000
    assert model._effective_bc_coefficient() == pytest.approx(0.0)


def _ranged_spacing_state():
    return {
        "arena": {"width": 1000, "height": 600},
        "player": {"position": {"x": 500, "y": 300}, "radius": 28},
        "combat": {
            "move_speed": 300,
            "weapon_range": 400,
            "weapon_count": 1,
            "ranged_count": 1,
            "melee_count": 0,
        },
        "enemies": [{
            "runtime_id": "enemy-1",
            "position": {"x": 680, "y": 300},
            "velocity": {"x": -100, "y": 0},
            "radius": 40,
            "attack_method": "contact",
        }],
        "projectiles": [],
        "attack_indicators": [],
        "projectile_paths": {},
    }


def test_ranged_spacing_prefers_distance_from_closing_enemy():
    state = _ranged_spacing_state()
    scorer = UnifiedHazardScorer(enabled=True)
    toward = scorer.risk_breakdown(state, MoveAction.RIGHT)
    away = scorer.risk_breakdown(state, MoveAction.LEFT)
    assert toward.ranged_spacing > away.ranged_spacing
    assert toward.total > away.total


def test_ranged_spacing_is_inactive_for_melee_or_missing_range():
    state = _ranged_spacing_state()
    scorer = UnifiedHazardScorer(enabled=True)
    state["combat"]["ranged_count"] = 0
    state["combat"]["melee_count"] = 1
    assert scorer.risk_breakdown(state, MoveAction.RIGHT).ranged_spacing == 0.0
    state["combat"]["ranged_count"] = 1
    state["combat"]["melee_count"] = 0
    state["combat"]["weapon_range"] = 0
    assert scorer.risk_breakdown(state, MoveAction.RIGHT).ranged_spacing == 0.0


def test_ranged_spacing_can_be_disabled_for_ablation():
    state = _ranged_spacing_state()
    scorer = UnifiedHazardScorer(enabled=True, ranged_spacing_enabled=False)
    assert scorer.risk_breakdown(state, MoveAction.RIGHT).ranged_spacing == 0.0
