"""Exercise the live environment contract without connecting to a game."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from brotato_ai.training.configs import load_config
from v4.env.brotato_api_env import BrotatoApiEnv


def state(tick, phase="combat", wave=1, **extra):
    return dict(
        session="outcome-test", tick=tick, published_at_ms=1000 + tick * 42,
        phase=phase, wave={"number": wave},
        arena={"width": 1920, "height": 1080},
        player={"position": {"x": 900, "y": 500}, "health": 10, "max_health": 10},
        enemies=[], projectiles=[], attack_indicators=[], pickups=[],
        counters={"materials": 0, "kills": 0}, **extra,
    )


class FakeBridge:
    def __init__(self, states):
        self.states = iter(states)

    def start(self): pass
    def close(self): pass
    def send(self, *args, **kwargs): pass
    def _write_final_action(self, *args, **kwargs): pass
    def mark_action_decision(self): pass
    def mark_action_sent(self): pass
    def mark_state_processing_start(self): pass
    def mark_state_processing_end(self): pass
    def wait_for_state(self, **kwargs): return deepcopy(next(self.states))


@pytest.mark.parametrize("phase,flags,expected", [
    ("game_over", {"dead": True}, (True, False, False, True, False)),
    ("victory", {"victory": True}, (False, True, True, True, False)),
    ("wave_end", {}, (False, False, True, False, True)),
    ("combat", {}, (False, False, False, False, False)),
])
def test_environment_returns_terminal_flags(phase, flags, expected):
    cfg = load_config({"BROTATO_V4_AUTOMATE_MENUS": "0", "BROTATO_V4_UI_DATASET": "off"})
    env = BrotatoApiEnv(cfg, server=FakeBridge([state(1), state(2, phase, **flags)]))
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(0)
        assert tuple(info[k] for k in ("dead", "victory", "wave_clear", "terminated", "truncated")) == expected
        assert (terminated, truncated) == expected[-2:]
    finally:
        env.close()


def test_wave_clear_survives_menu_automation_and_is_not_repeated():
    cfg = load_config({"BROTATO_V4_AUTOMATE_MENUS": "1", "BROTATO_V4_UI_DATASET": "off"})
    env = BrotatoApiEnv(cfg, server=FakeBridge([state(1), state(2, "wave_end"), state(5, wave=2)]))
    env.ui_controller.advance = lambda *args, **kwargs: SimpleNamespace(
        state=state(4, wave=2), sequence=3,
        states=[state(3, "shop"), state(4, wave=2)], sent_roles=[], confirmed_roles=[],
    )
    try:
        env.reset()
        _, _, terminal, truncated, info = env.step(0)
        assert info["wave_clear"] and info["wave"] == 2
        assert not terminal and not truncated
        assert not env.step(0)[-1]["wave_clear"]
    finally:
        env.close()
