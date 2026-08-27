import importlib


def test_active_boundaries_import_without_the_game():
    modules = [
        "brotato_ai.domain.state",
        "brotato_ai.bridge.client",
        "brotato_ai.control.hazards",
        "brotato_ai.control.arbiter",
        "brotato_ai.data.recorder",
        "brotato_ai.evaluation.backtest",
        "brotato_ai.training.configs",
    ]
    for name in modules:
        assert importlib.import_module(name) is not None

