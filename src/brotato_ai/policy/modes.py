"""Production policy modes.

- HANDCRAFTED (default): today's behavior, byte-identical.  No learned-policy
  code runs.
- SHADOW_HUMAN: the handcrafted path acts; the human policy predicts silently
  and the proposal is only logged.
- HYBRID_HUMAN: handcrafted decision timing selects when to decide; at a
  decision point the human action head proposes, persistence holds it in
  real time, and the existing safety arbiter stays the only override
  authority.
- EXPERIMENTAL_FULL_LEARNED: the human policy proposes every step and the
  change gate may be consulted.  Explicitly gated by configuration; never a
  default.
"""

from __future__ import annotations

from enum import Enum


class PolicyMode(str, Enum):
    HANDCRAFTED = "HANDCRAFTED"
    SHADOW_HUMAN = "SHADOW_HUMAN"
    HYBRID_HUMAN = "HYBRID_HUMAN"
    EXPERIMENTAL_FULL_LEARNED = "EXPERIMENTAL_FULL_LEARNED"


HANDCRAFTED = PolicyMode.HANDCRAFTED
SHADOW_HUMAN = PolicyMode.SHADOW_HUMAN
HYBRID_HUMAN = PolicyMode.HYBRID_HUMAN
EXPERIMENTAL_FULL_LEARNED = PolicyMode.EXPERIMENTAL_FULL_LEARNED

DEFAULT_POLICY_MODE = PolicyMode.HANDCRAFTED


def parse_policy_mode(value: str | PolicyMode | None) -> PolicyMode:
    """Parse a mode string; raises ``ValueError`` loudly on unknown values."""

    if value is None or str(value).strip() == "":
        return DEFAULT_POLICY_MODE
    if isinstance(value, PolicyMode):
        return value
    normalized = str(value).strip().upper().replace("-", "_")
    try:
        return PolicyMode(normalized)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in PolicyMode)
        raise ValueError(f"unknown policy mode {value!r}; expected one of: {valid}") from exc
