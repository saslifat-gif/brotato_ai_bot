"""Stable import surface for the build policy.

The implementation classes remain owned by ``v3.ui_build_policy`` (rule
teachers, learned policy, decision logger).  The mode contract lives in the
lightweight ``brotato_ai.ui.modes`` so configuration code never needs to
import the torch-heavy implementation; this facade re-exports both.
"""

from v3.ui_build_policy import *  # noqa: F401,F403 - compatibility facade

from brotato_ai.ui.modes import (  # noqa: F401
    DEFAULT_BUILD_POLICY_MODE,
    BuildPolicyMode,
    parse_build_policy_mode,
)
