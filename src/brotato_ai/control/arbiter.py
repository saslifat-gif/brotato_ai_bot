"""One action-resolution pipeline and one final bridge writer."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from brotato_ai.control.hazards import UnifiedHazardScorer
from brotato_ai.control.recovery import CrowdRecoveryGuard
from brotato_ai.domain.actions import MoveAction
from brotato_ai.domain.decisions import DecisionTrace, SafetyDecision
from brotato_ai.domain.state import StateSnapshot


class FinalActionTransport(Protocol):
    def _write_final_action(
        self, action: int, sequence: int, timeout_sec: float = 10.0
    ) -> None: ...


class FinalActionArbiter:
    """Resolve policy -> unified hazard -> explicit recovery -> final action."""

    def __init__(
        self,
        *,
        safety_shield: UnifiedHazardScorer,
        crowd_recovery_guard: CrowdRecoveryGuard,
    ):
        if crowd_recovery_guard.shield is not safety_shield:
            raise ValueError("crowd recovery must reuse the unified hazard scorer")
        self.safety_shield = safety_shield
        self.crowd_recovery_guard = crowd_recovery_guard

    def reset(self) -> None:
        self.crowd_recovery_guard.reset()

    def apply(
        self,
        state: Mapping[str, Any] | StateSnapshot,
        requested_action: int,
        *,
        previous_action: int | None = None,
        state_interval_ms: float = 0.0,
        control_interval_ms: float = 0.0,
    ) -> DecisionTrace:
        snapshot = StateSnapshot.from_payload(state)
        requested = int(MoveAction(int(requested_action)))
        risks = self.safety_shield.all_risks(snapshot)
        requested_risk = risks[requested]
        hazard_decision = self.safety_shield.choose(
            risks, requested, previous_action=previous_action
        )
        hazard_risk = risks[hazard_decision.applied_action]
        recovery_decision = self.crowd_recovery_guard.apply(
            snapshot,
            hazard_decision.applied_action,
            risks=risks,
            previous_action=previous_action,
            control_interval_ms=max(0.0, float(control_interval_ms)),
        )
        applied_risk = risks[recovery_decision.applied_action]
        decision = SafetyDecision(
            requested,
            recovery_decision.applied_action,
            requested_risk.total,
            applied_risk.total,
        )
        enemy_contact_overridden = bool(
            decision.overridden
            and requested_risk.enemy_total - applied_risk.enemy_total >= 0.08
        )
        recovery_active = self.crowd_recovery_guard.active
        source = (
            "crowd_recovery"
            if recovery_active
            else "hazard"
            if hazard_decision.overridden
            else "policy"
        )
        return DecisionTrace(
            decision=decision,
            hazard_decision=hazard_decision,
            recovery_decision=recovery_decision,
            requested_risk=requested_risk,
            hazard_risk=hazard_risk,
            applied_risk=applied_risk,
            source=source,
            recovery_active=recovery_active,
            enemy_contact_overridden=enemy_contact_overridden,
            session=snapshot.session,
            tick=snapshot.tick,
            timestamp_ms=snapshot.timestamp_ms,
            state_interval_ms=max(0.0, float(state_interval_ms)),
            control_interval_ms=max(0.0, float(control_interval_ms)),
            all_risks=dict(risks),
            tactical_state=self.crowd_recovery_guard.state_name,
            escape_remaining=int(self.crowd_recovery_guard.remaining),
            escape_side=int(self.crowd_recovery_guard.escape_side),
            escape_remaining_ms=float(getattr(self.crowd_recovery_guard, "remaining_ms", 0.0)),
        )


CombatDecisionPipeline = FinalActionArbiter


class FinalActionWriter:
    """The sole production owner allowed to emit a movement action."""

    def __init__(self, transport: FinalActionTransport, *, timeout_sec: float = 10.0):
        self.transport = transport
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.write_count = 0

    def write(self, trace: DecisionTrace, sequence: int) -> None:
        self.transport._write_final_action(
            trace.decision.applied_action,
            int(sequence),
            timeout_sec=self.timeout_sec,
        )
        self.write_count += 1
