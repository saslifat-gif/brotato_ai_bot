"""Learned human-policy integration layer.

This package isolates everything that touches the learned human movement
model.  The production controller must remain fully functional when none of
these objects are constructed; the only production seams are the optional
mode wiring in the environment and the optional DecisionTrace fields.
"""

from brotato_ai.policy.features import (
    EVENT_FEATURE_SCHEMA_VERSION,
    HumanPolicyFeatureBuilder,
)
from brotato_ai.policy.human_action import (
    EVENT_CHECKPOINT_FORMAT,
    EVENT_POLICY_SCHEMA_VERSION,
    EventHumanActionPolicy,
    EventHumanModel,
    HumanPolicyError,
    HumanProposal,
    load_event_checkpoint,
    save_event_checkpoint,
)
from brotato_ai.policy.hybrid import (
    DecisionTrigger,
    HumanHybridController,
    HybridResolution,
    PersistenceManager,
)
from brotato_ai.policy.modes import (
    EXPERIMENTAL_FULL_LEARNED,
    HANDCRAFTED,
    HYBRID_HUMAN,
    SHADOW_HUMAN,
    PolicyMode,
    parse_policy_mode,
)

__all__ = [
    "EVENT_FEATURE_SCHEMA_VERSION",
    "EVENT_CHECKPOINT_FORMAT",
    "EVENT_POLICY_SCHEMA_VERSION",
    "EventHumanActionPolicy",
    "EventHumanModel",
    "HumanPolicyError",
    "HumanProposal",
    "HumanPolicyFeatureBuilder",
    "DecisionTrigger",
    "HumanHybridController",
    "HybridResolution",
    "PersistenceManager",
    "EXPERIMENTAL_FULL_LEARNED",
    "HANDCRAFTED",
    "HYBRID_HUMAN",
    "SHADOW_HUMAN",
    "PolicyMode",
    "load_event_checkpoint",
    "save_event_checkpoint",
    "parse_policy_mode",
]
