"""Typed action-resolution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DECISION_SCHEMA_VERSION = 2
DecisionSource = Literal["policy", "hazard", "crowd_recovery", "material_pickup"]


@dataclass(frozen=True)
class SafetyDecision:
    requested_action: int
    applied_action: int
    requested_risk: float
    applied_risk: float

    @property
    def overridden(self) -> bool:
        return self.requested_action != self.applied_action


@dataclass(frozen=True)
class HazardRisk:
    enemy: float = 0.0
    projectile: float = 0.0
    indicator: float = 0.0
    boundary: float = 0.0
    enemy_path: float = 0.0
    projectile_path: float = 0.0
    boundary_path: float = 0.0
    ranged_spacing: float = 0.0

    @property
    def total(self) -> float:
        return float(
            self.enemy
            + self.projectile
            + self.indicator
            + self.boundary
            + self.enemy_path
            + self.projectile_path
            + self.boundary_path
            + self.ranged_spacing
        )

    @property
    def path(self) -> float:
        return float(self.enemy_path + self.projectile_path + self.boundary_path)

    @property
    def enemy_total(self) -> float:
        return float(self.enemy + self.enemy_path + self.ranged_spacing)

    @property
    def projectile_total(self) -> float:
        return float(self.projectile + self.projectile_path)

    @property
    def boundary_total(self) -> float:
        return float(self.boundary + self.boundary_path)

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "enemy": self.enemy_total,
            "projectile": self.projectile_total,
            "telegraph": self.indicator,
            "boundary": self.boundary_total,
            "enemy_geometry": self.enemy,
            "projectile_geometry": self.projectile,
            "boundary_geometry": self.boundary,
            "enemy_path": self.enemy_path,
            "projectile_path": self.projectile_path,
            "boundary_path": self.boundary_path,
            "ranged_spacing": self.ranged_spacing,
        }


@dataclass(frozen=True)
class DecisionTrace:
    """Exactly one policy request and one final movement action."""

    decision: SafetyDecision
    hazard_decision: SafetyDecision
    recovery_decision: SafetyDecision
    requested_risk: HazardRisk
    hazard_risk: HazardRisk
    applied_risk: HazardRisk
    source: DecisionSource
    recovery_active: bool
    enemy_contact_overridden: bool = False
    session: str = ""
    tick: int = -1
    timestamp_ms: int = -1
    state_interval_ms: float = 0.0
    control_interval_ms: float = 0.0
    all_risks: dict[int, HazardRisk] = field(default_factory=dict)
    tactical_state: str = "normal"
    escape_remaining: int = 0
    escape_remaining_ms: float = 0.0
    escape_side: int = 0
    anti_stall_active: bool = False
    idle_escape_ms: float = 0.0
    # Learned human-policy fields (schema 2).  They are None/"" whenever the
    # policy mode does not run the human model (HANDCRAFTED) or inference
    # failed; they never change action semantics by themselves.
    human_proposed_action: int | None = None
    human_confidence: float | None = None
    human_change_probability: float | None = None
    human_duration_ms: float | None = None
    human_source: str = ""
    human_used: bool = False
    human_agrees: bool | None = None
    human_fallback_reason: str = ""
    schema_version: int = DECISION_SCHEMA_VERSION

    @property
    def hazard_overridden(self) -> bool:
        return self.hazard_decision.overridden

    @property
    def recovery_overridden(self) -> bool:
        return self.recovery_decision.overridden

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_type": "decision_trace",
            "session": self.session,
            "tick": self.tick,
            "timestamp_ms": self.timestamp_ms,
            "requested_action": self.decision.requested_action,
            "final_action": self.decision.applied_action,
            "decision_source": self.source,
            "override": self.decision.overridden,
            "hazard_override": self.hazard_overridden,
            "recovery_mode": self.recovery_active,
            "tactical_state": self.tactical_state,
            "escape_remaining": int(self.escape_remaining),
            "escape_remaining_ms": float(self.escape_remaining_ms),
            "escape_side": int(self.escape_side),
            "anti_stall_active": self.anti_stall_active,
            "idle_escape_ms": self.idle_escape_ms,
            "action_risks": {str(a): risk.to_dict() for a, risk in self.all_risks.items()},
            "requested_risk": self.requested_risk.to_dict(),
            "applied_risk": self.applied_risk.to_dict(),
            "human_proposed_action": self.human_proposed_action,
            "human_confidence": self.human_confidence,
            "human_change_probability": self.human_change_probability,
            "human_duration_ms": self.human_duration_ms,
            "human_source": self.human_source,
            "human_used": self.human_used,
            "human_agrees": self.human_agrees,
            "human_fallback_reason": self.human_fallback_reason,
            "state_interval_ms": float(self.state_interval_ms),
            "control_interval_ms": float(self.control_interval_ms),
            "minimum_action_risk": min(
                (risk.total for risk in self.all_risks.values()), default=0.0
            ),
            "unsafe_action_count": sum(
                risk.total >= 0.65 for risk in self.all_risks.values()
            ),
            "requested_to_minimum_regret": max(
                0.0,
                self.requested_risk.total
                - min((risk.total for risk in self.all_risks.values()), default=self.requested_risk.total),
            ),
        }

