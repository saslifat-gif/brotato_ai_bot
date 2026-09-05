# Dangerous-idle escape

The wave-17 trace contained 110 idle actions in 130 decisions. Recovery chose a
moving escape, then its global minimum-risk fallback repeatedly restored idle.
Idle risk rose from zero to roughly 0.6 while eight moving choices were marked
unsafe. At the end, all nine actions were unsafe. The trace lacks positions and
full action scores, so it cannot establish a safe counterfactual or exact cause.

The recovery controller now tracks idle time. After 350 ms, if idle has enemy
risk of at least 0.16, it can select a moving direction that predicts at least
8 units of increased separation and points away from the nearest enemy.
The permitted total-risk increase is capped at 0.35; projectile and indicator
increases at 0.02 each, boundary increase at 0.05, and direct enemy-risk increase
at 0.10. If no candidate qualifies, idle remains permitted. This is a bounded
escape attempt, not a blanket ban on standing still or a guarantee of safety.

Moving and episode reset clear the idle timer. Timing uses measured intervals,
with the existing 24 Hz fallback when an interval is unavailable. Future decision
traces include the trigger, idle duration, and all nine action-risk breakdowns.

Validation: 215 Windows tests passed. Regression tests reproduce the indefinite
fallback pattern, verify time-based activation, and reject projectile, telegraph,
boundary and excessive enemy hazards. These are synthetic tests based on the
observed pattern, not a faithful replay of the recorded death. Live survival
improvement remains unmeasured. The already-running three-game batch retains its
previous in-memory controller for a consistent comparison.
