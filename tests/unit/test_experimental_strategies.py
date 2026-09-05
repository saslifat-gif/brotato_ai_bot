from copy import deepcopy
from types import SimpleNamespace

from brotato_ai.control.route_planner import EscapeRoutePlanner
from v4.experimental_shop import AdaptiveSmgTeacher
from v4.ui_build_policy import RangedSmgTeacher


def test_route_forecast_moves_entities_without_mutating_live_state():
    state = {"player": {"position": {"x": 100., "y": 100.}},
             "combat": {"move_speed": 200},
             "projectiles": [{"position": {"x": 500., "y": 100.}, "velocity": {"x": -100., "y": 0.}}],
             "projectile_paths": {"action_risk": [1] * 9}}
    before = deepcopy(state)
    forecast = EscapeRoutePlanner().forecast(state, 4)
    assert state == before
    assert forecast["projectiles"][0]["position"]["x"] == 455
    assert "projectile_paths" not in forecast
    assert forecast["player"]["position"] != state["player"]["position"]


def test_route_planner_rejects_a_short_term_safe_dead_end():
    planner = EscapeRoutePlanner()

    class Scorer:
        def all_risks(self, state):
            if "forecast_action" not in state:
                return {a: SimpleNamespace(total=0 if a in (3, 4) else 10) for a in range(9)}
            return {a: SimpleNamespace(total=2 if state["forecast_action"] == 4 else .1) for a in range(9)}

    planner.scorer = Scorer()
    planner.forecast = lambda state, action: {"forecast_action": action}
    action, trace = planner.propose({}, 4)
    assert action == 3 and trace["future_exits"][3] > trace["future_exits"][4]


def test_route_planner_preserves_request_without_a_material_gain():
    planner = EscapeRoutePlanner()
    planner.scorer = SimpleNamespace(all_risks=lambda state: {a: SimpleNamespace(total=0.) for a in range(9)})
    planner.forecast = lambda state, action: {}
    assert planner.propose({}, 4)[0] == 4


def test_experimental_shop_prioritizes_armor_more_when_build_lacks_it():
    teacher = AdaptiveSmgTeacher()
    choice = {"category": "item", "effects": [{"key": "stat_armor", "value": 2}]}
    low = {"build": {"stats": {"stat_armor": 0}}}
    high = {"build": {"stats": {"stat_armor": 30}}}
    assert teacher.score_choice(choice, 8, low) > teacher.score_choice(choice, 8, high)


def test_shop_experiment_does_not_change_baseline_weapon_allowance():
    assert RangedSmgTeacher._early_weapon_cap(2) == 1
    assert AdaptiveSmgTeacher._early_weapon_cap(2) == 2
