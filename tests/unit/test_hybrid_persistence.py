"""Decision trigger and real-time persistence semantics."""

import pytest

from brotato_ai.policy.human_action import HumanProposal
from brotato_ai.policy.hybrid import (
    DecisionTrigger,
    HumanHybridController,
    PersistenceManager,
)


def _proposal(action=6, probability=0.9, held=4):
    return HumanProposal(
        action=action,
        probability=probability,
        probabilities=np_eye_one_hot(action, probability),
        change_probability=0.5,
        duration_ms=438.0,
        held_action=held,
    )


def np_eye_one_hot(action, probability):
    import numpy as np

    values = np.zeros(9, dtype=np.float64)
    values[action] = probability
    return values


class TestDecisionTrigger:
    def test_first_call_and_escape_are_decision_points(self):
        trigger = DecisionTrigger(decision_interval_ms=438.0)
        assert trigger.should_decide(escape_active=False, now_ms=0.0) is True
        trigger.mark_decision(0.0)
        assert trigger.should_decide(escape_active=False, now_ms=100.0) is False
        assert trigger.should_decide(escape_active=True, now_ms=100.0) is True

    def test_interval_is_real_time_not_steps(self):
        trigger = DecisionTrigger(decision_interval_ms=438.0)
        trigger.mark_decision(0.0)
        # At 24 Hz, 438 ms spans ~10.5 steps; at 60 Hz it spans ~26 steps.
        # The trigger must behave identically in both cases.
        assert trigger.should_decide(escape_active=False, now_ms=437.9) is False
        assert trigger.should_decide(escape_active=False, now_ms=438.1) is True


class TestPersistenceManager:
    def test_hold_expires_after_real_time_prior(self):
        persistence = PersistenceManager(hold_prior_ms=438.0)
        persistence.hold(6, now_ms=0.0)
        assert persistence.current(now_ms=200.0) == 6
        assert persistence.current(now_ms=437.9) == 6
        assert persistence.current(now_ms=438.1) is None
        assert persistence.remaining_ms(now_ms=100.0) == pytest.approx(338.0)

    def test_release_and_rehold(self):
        persistence = PersistenceManager(hold_prior_ms=438.0)
        persistence.hold(2, now_ms=0.0)
        persistence.release()
        assert persistence.current(now_ms=10.0) is None
        persistence.hold(8, now_ms=20.0)
        assert persistence.current(now_ms=30.0) == 8


class TestHumanHybridController:
    def test_persistence_holds_between_decision_points(self):
        controller = HumanHybridController(
            decision_interval_ms=438.0, hold_prior_ms=438.0, min_confidence=0.3
        )
        resolution = controller.resolve(
            requested_action=2, escape_active=False, proposal=_proposal(action=6), now_ms=0.0
        )
        assert resolution.used_human and resolution.source == "human_trigger"
        # Well before the next decision interval, persistence wins even when
        # the handcrafted controller asks for something else.
        held = controller.resolve(
            requested_action=2, escape_active=False, proposal=_proposal(action=8), now_ms=100.0
        )
        assert held.requested_action == 6 and held.source == "human_persistence"

    def test_escape_releases_hold_immediately(self):
        controller = HumanHybridController(
            decision_interval_ms=438.0, hold_prior_ms=438.0, min_confidence=0.3
        )
        controller.resolve(
            requested_action=2, escape_active=False, proposal=_proposal(action=6), now_ms=0.0
        )
        resolution = controller.resolve(
            requested_action=3, escape_active=True, proposal=_proposal(action=8), now_ms=50.0
        )
        assert resolution.requested_action == 8 and resolution.reason == "escape_trigger"

    def test_low_confidence_falls_back_to_handcrafted(self):
        controller = HumanHybridController(
            decision_interval_ms=438.0, hold_prior_ms=438.0, min_confidence=0.9
        )
        resolution = controller.resolve(
            requested_action=5, escape_active=False, proposal=_proposal(probability=0.4), now_ms=0.0
        )
        assert resolution.requested_action == 5 and not resolution.used_human

    def test_missing_proposal_falls_back_to_handcrafted(self):
        controller = HumanHybridController(min_confidence=0.3)
        resolution = controller.resolve(
            requested_action=5, escape_active=False, proposal=None, now_ms=0.0
        )
        assert resolution.requested_action == 5 and not resolution.used_human

    def test_full_learned_bypasses_persistence(self):
        controller = HumanHybridController(full_learned=True, min_confidence=0.3)
        first = controller.resolve(
            requested_action=2, escape_active=False, proposal=_proposal(action=6), now_ms=0.0
        )
        assert first.requested_action == 6
        second = controller.resolve(
            requested_action=2, escape_active=False, proposal=None, now_ms=10.0
        )
        assert second.requested_action == 2 and second.source == "handcrafted"

    def test_zero_confidence_accepts_any_proposal(self):
        controller = HumanHybridController(min_confidence=0.0)
        resolution = controller.resolve(
            requested_action=2, escape_active=False, proposal=_proposal(probability=0.12), now_ms=0.0
        )
        assert resolution.used_human
