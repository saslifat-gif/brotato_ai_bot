# Active V4 tactical movement

The active V4 movement path has one action owner:

policy request
  -> UnifiedHazardScorer (all candidate risks)
  -> TacticalMovementController (temporal state + escape choice)
  -> FinalActionWriter (one bridge write)

CrowdRecoveryGuard remains as a compatibility name for
TacticalMovementController; it is not a second controller.

## State machine

- NORMAL: the policy/hazard result is used unless a meaningful threat or
  legacy crowd/boundary emergency starts an escape.
- ESCAPE: the controller holds for hold_steps, keeps a persistent lateral
  side, penalizes movement toward the predicted enemy position, and penalizes
  dangerous reversals.
- Release requires the minimum hold, low requested risk, and either no active
  enemy or predicted separation at least release_margin (1.15) above the
  target band. A one-frame risk drop cannot release escape.

The same enemy_separation_diagnostics function is used by hazard scoring,
ranged-spacing scoring, tactical escape, and replay evaluation. It models enemy
velocity over the 450 ms horizon and returns predicted separation, closing rate,
and radial movement alignment.

Ranged combat keeps OBJECTIVE_ENGAGE as the combat objective but sends the
movement target to a stand-off point or tangent/orbit waypoint. The movement
target is therefore not the enemy center.

## Replay evidence

The latest 2,000-record fixed replay produced this geometric A/B:

| Metric | Baseline unified | Persistent tactical |
| --- | ---: | ---: |
| Escape entries | 61 | 32 |
| Escape direction reversals | 37 | 22 |
| Mean enemy separation | 210.3 | 226.8 |
| Post-escape re-entry rate | 91.9% | 0.0% |

The persistent controller used more escape steps and more total overrides. This
is an intentional safety trade-off that must be checked against live damage,
survival, and victory metrics.

The fixed raw recording contains empty combat metadata, so it cannot validate
ranged-spacing activation. An enriched 40-record fixture did activate the
spacing scorer: spacing-on modeled risk was 0.0542 versus 0.0 with spacing
off, and it reduced modeled direction switches from 3 to 0. These are
geometric counterfactuals, not proof of alternate game outcomes.

## Live metrics

After restarting the trainer, inspect:

- combat/tactical_escape
- combat/tactical_escape_remaining
- combat/tactical_escape_side
- combat/tactical_state_entry
- combat/ranged_spacing_active
- combat/ranged_spacing_dist
- combat/ranged_spacing_closing
- combat/hazard_applied_enemy_risk
- combat/projectile_path_max_risk

For acceptance, compare wave survival, damage, combat/best_wave, and
combat/victory against a run from the prior active checkout. A healthy escape
controller should not remain in escape permanently, should not hug the arena
boundary, and should reduce post-escape contact/re-entry events.
