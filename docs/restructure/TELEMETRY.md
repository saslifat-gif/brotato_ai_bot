
# V4 Runtime Telemetry Contract

This document defines the stable metrics added by the review pass. Unless noted
otherwise, callback values named *_rate or *_share are dump-window means.

## Outcomes

- `combat/best_wave`, `combat/current_wave`, and
  `combat/episode_wave` are observed wave numbers.
- `combat/episode_death`, `combat/episode_victory`, and
  `combat/episode_wave_clear` are terminal/event labels.
- `combat/death_count_total`, `combat/victory_count_total`, and
  `combat/wave_clear_count_total` are process-local counters.
- Wave death tags count only explicit `dead=true`; victory is never a death.

## Actions and hazard selection

- `actions/requested_0..8` are policy request shares.
- `actions/applied_0..8` are final action shares.
- `actions/requested_applied_disagreement` is the request/final mismatch rate.
- `combat/hazard_requested_risk` and `combat/hazard_applied_risk` are the
  unitless unified heuristic score, not collision probabilities.
- `combat/min_action_risk`, `combat/unsafe_action_count`,
  `combat/unsafe_action_fraction`, and
  `combat/requested_to_minimum_regret` describe all nine actions. Unsafe means
  total score >= 0.65.
- The fixed replay report compares `policy_only` (shield off) with
  `unified` (shield on) on exactly the same recording. It reports modeled
  risk only; it cannot prove alternate game outcomes.

## Timing and transport

- `control/effective_state_hz` is the EMA of instantaneous published-state
  rates: `f_t = 1000/delta_t`, then
  `f_t = 0.9 f_(t-1) + 0.1 f_t`.
- `control/state_interval_p50_ms`, `p95`, and `p99`, plus the control
  equivalents, expose tail latency.
- `control/stale_state_count` counts states rejected for an old tick or
  sequence; `control/dropped_state_count` includes stale states and tick gaps.
- `control/state_tick_gap` is the latest accepted tick gap.
- `control/reward_time_scale = clip(delta_t * reference_hz / 1000, 0.25, 4)`.
  Only dense survival/path/motion shaping is scaled; wave, death, and victory
  events are not.

## Exposure and path diagnostics

- TTI and miss distance use -1 only when no hostile projectile exists.
  Conditional metrics use only valid exposures and never average sentinel values.
- `combat/proj_tti_exp_rate` and
  `combat/proj_hazard_exp_rate` identify conditional
  hazard windows; conditional damage and miss-distance tags are emitted
  separately.
- Path metrics are split into pre-action and post-action fields in the
  environment info. Existing `*_path_action_risk` tags remain for compatibility.
  Projectile, enemy, and boundary path risk are scored by the same unified
  selector.

## Temporal and supervised validation

- `v4/temporal_residual_logit_norm`,
  `v4/temporal_policy_disagreement`, and
  `v4/temporal_legacy_final_kl` show whether the temporal residual is active.
- `tools/inspect/temporal_ablation.py` evaluates identical inputs with
  normal, zeroed-history, and deterministically shuffled-history variants.
- Human-anchor training uses a deterministic 90/10 split and logs overall plus
  per-action held-out validation accuracy under `human_bc/validation_*`.

## Acceptance sequence

After a normal trainer stop and game restart, verify the bridge handshake,
then start the scheduled trainer. Confirm the new tags exist in the newest
TensorBoard run. Use fixed replay first, then compare multiple live seeds;
never claim improvement from one live run alone.

