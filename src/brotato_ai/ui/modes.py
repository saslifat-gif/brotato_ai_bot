"""Build-policy mode contract (lightweight; safe to import from config).

Build decisions run on a different timescale from movement and are selected
through an explicit mode (spec section 11):

- HANDCRAFTED (default): the rule teachers in ``v3.ui_build_policy`` decide,
  exactly as in production today.
- HUMAN_RECORDED: human choices are captured through the decision logger for
  later study; selection stays handcrafted until a validated model exists.
- LEARNED: a build-choice model refines the teacher-gated ranking.  It is
  refused by configuration validation unless the model path was set
  explicitly, so an undertrained candidate checkpoint can never be
  auto-deployed by output-directory discovery.
"""

from __future__ import annotations

from enum import Enum


class BuildPolicyMode(str, Enum):
    HANDCRAFTED = "HANDCRAFTED"
    HUMAN_RECORDED = "HUMAN_RECORDED"
    LEARNED = "LEARNED"


DEFAULT_BUILD_POLICY_MODE = BuildPolicyMode.HANDCRAFTED


def parse_build_policy_mode(value: str | BuildPolicyMode | None) -> BuildPolicyMode:
    if value is None or str(value).strip() == "":
        return DEFAULT_BUILD_POLICY_MODE
    if isinstance(value, BuildPolicyMode):
        return value
    normalized = str(value).strip().upper().replace("-", "_")
    try:
        return BuildPolicyMode(normalized)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in BuildPolicyMode)
        raise ValueError(f"unknown build policy mode {value!r}; expected one of: {valid}") from exc
