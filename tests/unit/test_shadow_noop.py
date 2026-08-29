"""Shadow mode must never change gameplay; trace fields must stay additive."""

import dataclasses

import pytest

from brotato_ai.domain.actions import MoveAction
from brotato_ai.domain.decisions import DECISION_SCHEMA_VERSION, DecisionTrace, SafetyDecision


def _trace(requested=3, applied=3):
    from brotato_ai.domain.decisions import HazardRisk

    risk = HazardRisk()
    return DecisionTrace(
        decision=SafetyDecision(requested, applied, risk.total, risk.total),
        hazard_decision=SafetyDecision(requested, applied, risk.total, risk.total),
        recovery_decision=SafetyDecision(requested, applied, risk.total, risk.total),
        requested_risk=risk,
        hazard_risk=risk,
        applied_risk=risk,
        source="policy",
        recovery_active=False,
    )


def test_decision_trace_schema_v2_has_additive_human_fields():
    trace = _trace()
    payload = trace.to_dict()
    assert trace.schema_version == DECISION_SCHEMA_VERSION == 2
    assert payload["human_proposed_action"] is None
    assert payload["human_used"] is False
    assert payload["human_source"] == ""
    # Schema-1 keys must remain present with unchanged meanings.
    for legacy_key in (
        "requested_action", "final_action", "decision_source", "override",
        "requested_risk", "applied_risk", "state_interval_ms",
    ):
        assert legacy_key in payload


def test_replace_attaches_human_fields_without_touching_safety_semantics():
    trace = _trace(requested=2, applied=7)
    updated = dataclasses.replace(
        trace,
        human_proposed_action=8,
        human_confidence=0.71,
        human_source="shadow",
        human_used=False,
    )
    assert updated.decision.requested_action == 2
    assert updated.decision.applied_action == 7
    assert updated.to_dict()["human_proposed_action"] == 8
    assert updated.to_dict()["decision_source"] == "policy"


class _StubBuilder:
    def __init__(self, action):
        self.action = action
        self.observed = 0

    def observe(self, state, held_action, *, timestamp_ms=None):
        self.observed += 1
        return None

    def build_input(self, held_action):
        import numpy as np

        return np.zeros(33, dtype=np.float32)

    def reset(self):
        pass


class _StubPolicy:
    def __init__(self, action=8, probability=0.9):
        self.action = action
        self.probability = probability

    def propose(self, model_input, held_action):
        from brotato_ai.policy.human_action import HumanProposal
        import numpy as np

        probabilities = np.zeros(9, dtype=np.float64)
        probabilities[self.action] = self.probability
        return HumanProposal(
            action=self.action,
            probability=self.probability,
            probabilities=probabilities,
            change_probability=0.25,
            duration_ms=438.0,
            held_action=held_action,
        )


def _bare_env(policy_mode, human_policy=None, human_builder=None, hybrid=None):
    pytest.importorskip("gymnasium")
    from v3.env.brotato_api_env import BrotatoApiEnv

    env = object.__new__(BrotatoApiEnv)
    from brotato_ai.performance import RuntimeProfiler

    env.profiler = RuntimeProfiler(enabled=False)
    env.policy_mode = policy_mode
    env.human_policy = human_policy
    env.human_builder = human_builder
    env.hybrid_controller = hybrid
    env.previous_action = int(MoveAction.LEFT)
    env.last_state = {}
    return env


def test_handcrafted_mode_is_a_no_op():
    env = _bare_env(__import__("brotato_ai.policy.modes", fromlist=["PolicyMode"]).PolicyMode.HANDCRAFTED)
    fields, effective = env._apply_human_policy(5, escape_active=False)
    assert fields == {} and effective == 5


def test_shadow_mode_records_but_never_changes_the_requested_action():
    from brotato_ai.policy.modes import PolicyMode

    builder = _StubBuilder(action=8)
    env = _bare_env(
        PolicyMode.SHADOW_HUMAN,
        human_policy=_StubPolicy(action=8),
        human_builder=builder,
    )
    fields, effective = env._apply_human_policy(5, escape_active=False)
    assert effective == 5, "shadow mode must not alter the requested action"
    assert fields["human_used"] is False
    assert fields["human_source"] == "shadow"
    assert fields["human_proposed_action"] == 8
    assert fields["human_agrees"] is False
    assert builder.observed == 1


def test_shadow_mode_survives_policy_failure():
    from brotato_ai.policy.modes import PolicyMode

    class _BrokenPolicy:
        def propose(self, model_input, held_action):
            raise RuntimeError("boom")

    env = _bare_env(
        PolicyMode.SHADOW_HUMAN,
        human_policy=_BrokenPolicy(),
        human_builder=_StubBuilder(action=8),
    )
    fields, effective = env._apply_human_policy(5, escape_active=False)
    assert effective == 5
    assert fields["human_proposed_action"] is None
    assert fields["human_fallback_reason"] == "no_proposal"


def test_hybrid_mode_replaces_requested_only_before_the_arbiter():
    from brotato_ai.policy.hybrid import HumanHybridController
    from brotato_ai.policy.modes import PolicyMode

    env = _bare_env(
        PolicyMode.HYBRID_HUMAN,
        human_policy=_StubPolicy(action=8),
        human_builder=_StubBuilder(action=8),
        hybrid=HumanHybridController(min_confidence=0.3),
    )
    fields, effective = env._apply_human_policy(5, escape_active=False)
    assert effective == 8
    assert fields["human_used"] is True
    assert fields["human_source"] == "human_trigger"
