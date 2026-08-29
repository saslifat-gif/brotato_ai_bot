"""Decision trigger, real-time persistence, and hybrid resolution.

Separation of concerns (spec sections 6 and 13):

- "when to decide" comes from the handcrafted tactical context plus a
  real-time minimum decision interval, NOT from the learned change gate.
- "what action to choose" comes from the human action head at those decision
  points.
- persistence holds the chosen action in milliseconds; the learned duration
  head stays a diagnostic and is never used for production timing.
- the output is only a *requested* action: the existing unified hazard
  arbiter remains the single override authority before anything is written
  to the bridge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from brotato_ai.policy.human_action import HumanProposal


DEFAULT_HOLD_MS = 438.0  # observed mean human action hold


def _now_ms() -> float:
    return time.monotonic_ns() / 1e6


class DecisionTrigger:
    """Decide when a meaningful movement decision point exists.

    A decision point is triggered by the handcrafted tactical controller
    entering escape, or when at least ``decision_interval_ms`` of real time
    has passed since the last decision.  This deliberately ignores the
    learned change gate (held-out change F1 ~0.14; experimental only).
    """

    def __init__(self, *, decision_interval_ms: float = DEFAULT_HOLD_MS):
        if decision_interval_ms < 0.0:
            raise ValueError("decision_interval_ms must be non-negative")
        self.decision_interval_ms = float(decision_interval_ms)
        self._last_decision_ms: float | None = None

    def reset(self) -> None:
        self._last_decision_ms = None

    def should_decide(self, *, escape_active: bool, now_ms: float | None = None) -> bool:
        now = _now_ms() if now_ms is None else float(now_ms)
        if escape_active or self._last_decision_ms is None:
            return True
        return (now - self._last_decision_ms) >= self.decision_interval_ms

    def mark_decision(self, now_ms: float | None = None) -> None:
        self._last_decision_ms = _now_ms() if now_ms is None else float(now_ms)


class PersistenceManager:
    """Human-style action persistence expressed in real time.

    The manager never blindly holds for the full prior when the tactical
    trigger fires: an escape decision point releases the hold immediately.
    """

    def __init__(self, *, hold_prior_ms: float = DEFAULT_HOLD_MS):
        if hold_prior_ms < 0.0:
            raise ValueError("hold_prior_ms must be non-negative")
        self.hold_prior_ms = float(hold_prior_ms)
        self._held_action: int | None = None
        self._held_since_ms: float | None = None

    def reset(self) -> None:
        self._held_action = None
        self._held_since_ms = None

    @property
    def held_action(self) -> Optional[int]:
        return self._held_action

    def hold(self, action: int, *, now_ms: float | None = None) -> None:
        self._held_action = int(action)
        self._held_since_ms = _now_ms() if now_ms is None else float(now_ms)

    def remaining_ms(self, now_ms: float | None = None) -> float:
        if self._held_since_ms is None:
            return 0.0
        now = _now_ms() if now_ms is None else float(now_ms)
        return max(0.0, self.hold_prior_ms - (now - self._held_since_ms))

    def current(self, now_ms: float | None = None) -> Optional[int]:
        """Return the persisted action while its real-time hold is active."""

        if self._held_action is None or self.remaining_ms(now_ms) <= 0.0:
            return None
        return self._held_action

    def release(self) -> None:
        self._held_action = None
        self._held_since_ms = None


@dataclass(frozen=True)
class HybridResolution:
    """Outcome of the hybrid controller for one control step."""

    requested_action: int
    source: str  # "handcrafted" | "human_trigger" | "human_persistence"
    used_human: bool
    reason: str
    persistence_remaining_ms: float


class HumanHybridController:
    """Resolve a requested action against the human proposal and persistence.

    Resolution order (HYBRID_HUMAN):

    1. while persistence holds and no escape decision point exists, keep the
       persisted human action;
    2. at a decision point, accept the human proposal when its confidence
       clears ``min_confidence`` and persist it;
    3. otherwise use the handcrafted requested action.

    EXPERIMENTAL_FULL_LEARNED bypasses persistence and the trigger: the
    proposal (if any) is always the requested action.
    """

    def __init__(
        self,
        *,
        decision_interval_ms: float = DEFAULT_HOLD_MS,
        hold_prior_ms: float = DEFAULT_HOLD_MS,
        min_confidence: float = 0.0,
        full_learned: bool = False,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        self.trigger = DecisionTrigger(decision_interval_ms=decision_interval_ms)
        self.persistence = PersistenceManager(hold_prior_ms=hold_prior_ms)
        self.min_confidence = float(min_confidence)
        self.full_learned = bool(full_learned)

    def reset(self) -> None:
        self.trigger.reset()
        self.persistence.reset()

    def resolve(
        self,
        *,
        requested_action: int,
        escape_active: bool,
        proposal: HumanProposal | None,
        now_ms: float | None = None,
    ) -> HybridResolution:
        requested = int(requested_action)
        now = _now_ms() if now_ms is None else float(now_ms)
        if self.full_learned:
            if proposal is not None:
                self.trigger.mark_decision(now)
                return HybridResolution(
                    proposal.action, "human_trigger", True, "full_learned_proposal", 0.0
                )
            return HybridResolution(
                requested, "handcrafted", False, "full_learned_without_proposal", 0.0
            )
        decision_point = self.trigger.should_decide(escape_active=escape_active, now_ms=now)
        if not decision_point:
            held = self.persistence.current(now)
            if held is not None:
                return HybridResolution(
                    held,
                    "human_persistence",
                    True,
                    "persistence_hold",
                    self.persistence.remaining_ms(now),
                )
        if proposal is not None and proposal.probability >= self.min_confidence:
            self.trigger.mark_decision(now)
            self.persistence.hold(proposal.action, now_ms=now)
            reason = "escape_trigger" if escape_active else "decision_interval"
            return HybridResolution(
                proposal.action, "human_trigger", True, reason, self.persistence.hold_prior_ms
            )
        if decision_point:
            self.trigger.mark_decision(now)
            self.persistence.release()
        return HybridResolution(
            requested, "handcrafted", False, "no_confident_proposal", 0.0
        )
