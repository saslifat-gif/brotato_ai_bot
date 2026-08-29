"""Policy-mode parsing and configuration validation matrix."""

from pathlib import Path

import pytest

from brotato_ai.policy.modes import (
    EXPERIMENTAL_FULL_LEARNED,
    HANDCRAFTED,
    HYBRID_HUMAN,
    SHADOW_HUMAN,
    parse_policy_mode,
)
from brotato_ai.training.configs import load_config


def test_parse_policy_mode_accepts_documented_names():
    assert parse_policy_mode(None) is HANDCRAFTED
    assert parse_policy_mode("") is HANDCRAFTED
    assert parse_policy_mode("HANDCRAFTED") is HANDCRAFTED
    assert parse_policy_mode("shadow-human") is SHADOW_HUMAN
    assert parse_policy_mode(" hybrid_human ") is HYBRID_HUMAN
    assert parse_policy_mode("experimental_full_learned") is EXPERIMENTAL_FULL_LEARNED


def test_parse_policy_mode_fails_loudly_on_unknown():
    with pytest.raises(ValueError, match="unknown policy mode"):
        parse_policy_mode("AUTOPILOT")


def _env(**overrides):
    env = {
        "BROTATO_V3_HOST": "127.0.0.1",
        "BROTATO_V3_PORT": "4242",
        "BROTATO_V3_OUTPUT_DIR": "models/version_3",
    }
    env.update(overrides)
    return env


def test_default_mode_is_handcrafted_without_model():
    config = load_config(_env())
    assert config.policy_mode is HANDCRAFTED
    assert config.human_model_path is None


def test_shadow_mode_requires_model_path():
    with pytest.raises(ValueError, match="human model path"):
        load_config(_env(BROTATO_V4_POLICY_MODE="SHADOW_HUMAN"))
    config = load_config(_env(
        BROTATO_V4_POLICY_MODE="SHADOW_HUMAN",
        BROTATO_V4_HUMAN_MODEL="models/version_3/event_human.pt",
    ))
    assert config.policy_mode is SHADOW_HUMAN
    assert config.human_model_path == Path("models/version_3/event_human.pt").resolve()


def test_full_learned_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit opt-in"):
        load_config(_env(
            BROTATO_V4_POLICY_MODE="EXPERIMENTAL_FULL_LEARNED",
            BROTATO_V4_HUMAN_MODEL="m.pt",
        ))
    with pytest.raises(ValueError, match="human model path"):
        load_config(_env(
            BROTATO_V4_POLICY_MODE="EXPERIMENTAL_FULL_LEARNED",
            BROTATO_V4_ALLOW_FULL_LEARNED="1",
        ))
    # The opt-in flag alone never changes the default mode.
    assert load_config(_env(BROTATO_V4_ALLOW_FULL_LEARNED="1")).policy_mode is HANDCRAFTED
    config = load_config(_env(
        BROTATO_V4_POLICY_MODE="EXPERIMENTAL_FULL_LEARNED",
        BROTATO_V4_HUMAN_MODEL="m.pt",
        BROTATO_V4_ALLOW_FULL_LEARNED="1",
    ))
    assert config.policy_mode is EXPERIMENTAL_FULL_LEARNED


def test_hybrid_mode_takes_realtime_timing_parameters():
    config = load_config(_env(
        BROTATO_V4_POLICY_MODE="HYBRID_HUMAN",
        BROTATO_V4_HUMAN_MODEL="m.pt",
        BROTATO_V4_HUMAN_HOLD_MS="500",
        BROTATO_V4_HUMAN_INTERVAL_MS="450",
        BROTATO_V4_HUMAN_CONFIDENCE="0.55",
    ))
    assert config.human_hold_prior_ms == 500.0
    assert config.human_decision_interval_ms == 450.0
    assert config.human_confidence_threshold == 0.55


def test_invalid_mode_and_thresholds_fail_validation():
    with pytest.raises(ValueError, match="unknown policy mode"):
        load_config(_env(BROTATO_V4_POLICY_MODE="WILD"))
    with pytest.raises(ValueError, match="human_confidence_threshold"):
        load_config(_env(BROTATO_V4_HUMAN_CONFIDENCE="1.5"))
    with pytest.raises(ValueError, match="at least 50 ms"):
        load_config(_env(BROTATO_V4_HUMAN_HOLD_MS="10"))


def test_startup_summary_includes_mode():
    config = load_config(_env(
        BROTATO_V4_POLICY_MODE="SHADOW_HUMAN",
        BROTATO_V4_HUMAN_MODEL="m.pt",
    ))
    summary = config.startup_summary()
    assert "policy_mode=SHADOW_HUMAN" in summary
    assert "human_model=" in summary
