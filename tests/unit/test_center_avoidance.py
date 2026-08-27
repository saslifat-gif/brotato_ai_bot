from v3.combat_policy import center_stagnation_signal


def _state(x: float, y: float, *, phase: str = "combat") -> dict:
    return {
        "phase": phase,
        "arena": {"width": 1000.0, "height": 1000.0},
        "player": {"position": {"x": x, "y": y}},
    }


def test_center_stagnation_requires_both_samples_inside_center_radius() -> None:
    assert center_stagnation_signal(
        _state(500.0, 500.0),
        _state(520.0, 500.0),
        threat_risk=0.0,
    )
    assert not center_stagnation_signal(
        _state(500.0, 500.0),
        _state(700.0, 500.0),
        threat_risk=0.0,
    )


def test_center_stagnation_is_disabled_when_threat_is_real() -> None:
    assert not center_stagnation_signal(
        _state(500.0, 500.0),
        _state(520.0, 500.0),
        threat_risk=0.45,
    )


def test_center_stagnation_requires_combat_phase() -> None:
    assert not center_stagnation_signal(
        _state(500.0, 500.0, phase="shop"),
        _state(520.0, 500.0, phase="shop"),
        threat_risk=0.0,
    )
