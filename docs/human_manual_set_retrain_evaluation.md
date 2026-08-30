# Human manual-set retrain and shadow evaluation

Date: 2026-08-29

Decision: **keep `HYBRID_HUMAN` disabled**. The new model is not a safe or
stable hybrid candidate.

## Dataset and capture quality

The three canonical recordings are on Windows under
`models/version_3/human_demos/`:

| Run | Build signature | Rich frames | Combat frames | Genuine transitions | Build snapshots | Outcome | Max wave |
|---|---|---:|---:|---:|---:|---|---:|
| 01 | `weapon_ghost_flint` | 19,891 | 14,193 | 2,464 | 5,698 | death | 20 |
| 02 | `weapon_double_barrel_shotgun` | 35,510 | 24,715 | 2,871 | 10,795 | victory | 20 |
| 03 | `weapon_wrench` | 34,813 | 25,570 | 3,227 | 9,243 | victory | 20 |
| **Total** | **3 non-SMG signatures** | **90,214** | **64,478** | **8,562** | **25,736** | **2 victories / 1 death** | **20** |

The merged capture has 219,127 raw samples. SQLite integrity, blob decoding,
episode boundaries, raw-sample assignment, input/reward/outcome coverage, and
training/live feature parity all pass. Rich timestamp drift is p90 11 ms
(maximum 203 ms); raw drift is p90 12 ms (maximum 271 ms). The remaining
265–453 ms frame-gap warnings are publication stalls, not missing or corrupt
streams.

Coverage includes 17,319 combat frames in waves 8–12, 43,401 dense-projectile
frames, 27,418 dense-enemy frames, 3,092 frames below 50% HP, and 59,211
recovery frames. There are 43 unique action-transition pairs with 4.47 bits of
transition entropy. The set is suitable for offline event-policy retraining,
although it does not yet include an SMG build.

## Retrained event model

Checkpoint: `models/version_3/human_demos/human_event_bc_manual_set.pt`

The primary complete-episode holdout (the shotgun run) produced:

- change F1: **0.0826**
- next-action accuracy on true changes: **73.8%**
- teacher-forced transition timing MAE: **165.8 ms**
- autoregressive action accuracy: **20.3%**
- autoregressive change F1: **0.0538**
- selected-action confidence ECE: **0.183**

Leave-one-episode-out results were inconsistent: ghost flint F1/next-action/
autoregressive accuracy = 0.300/83.0%/36.9%; wrench = 0.043/69.6%/11.8%;
shotgun = 0.083/73.8%/20.3%. The mean autoregressive action accuracy was
23.0%. The high-confidence action bin averaged 0.978 confidence but only
82.0% accuracy, so calibration is still overconfident.

## SHADOW_HUMAN against the repaired PPO baseline

Baseline: `models/version_3/human_base_ppo_recovery.zip`, verified as a
non-collapsed PPO with the existing safety stack. Three complete shadow
episodes were run with the applied action unchanged by shadow mode; all ended
at wave 2, so this live batch does not establish late-wave performance.

| Metric | Previous non-collapsed model | New manual-set model |
|---|---:|---:|
| Proposal agreement | 4.53% | **1.07%** |
| Actual applied-action safety override | 42.57% | 30.25% |
| Human proposal counterfactual override | 79.80% | **96.15%** |
| Higher-risk proposal rate | 70.88% | **86.07%** |
| High-confidence disagreement rate | 95.79% | **99.07%** |
| Offline selected-action accuracy | 87.36% | 73.76% |
| Offline action ECE | 0.029 | **0.183** |

The safety and risk columns are geometric/counterfactual diagnostics, not
proof of alternate game outcomes. They nevertheless point consistently in the
wrong direction for the new model.

## Manual high-confidence review

Representative records from the new shadow log:

- Wave 1, tick 524: confidence 0.738, proposal `UP_LEFT` versus handcrafted
  `LEFT`, with proposal risk 17.34 versus 8.26 in a 4-enemy/3-projectile/
  3-telegraph state.
- Wave 1, tick 549: confidence 0.938, proposal `UP_LEFT` versus `LEFT`, with
  risk 23.17 versus 7.83 in a dense projectile state.
- Wave 2, tick 1158: confidence 0.779, proposal `UP` versus `UP_RIGHT`, at
  39% HP and 12 enemies; proposal risk exceeded handcrafted risk by 10.31.
- Wave 2, tick 1228: confidence 0.985, proposal `DOWN_LEFT` versus `DOWN`,
  at 4.3% HP and 12 enemies; the realized episode died on the next tick.
- A smaller set of proposals did have lower modeled risk, such as wave 1,
  tick 589 (confidence 0.742; proposal risk 7.82 versus handcrafted 16.44),
  confirming that the comparison is mixed at individual frames rather than a
  universal direction bias.

These examples were reviewed as state/action records. Because the proposal was
never applied in SHADOW_HUMAN, the observed damage/death cannot be attributed
to it; the high-risk result is a counterfactual safety warning.

## Recommendation

Do not enable `HYBRID_HUMAN`. The new model improves the raw next-action
signal over persistence but does not improve stable event timing or
autoregressive generalization, is overconfident, and is more frequently marked
as unsafe than both prior shadow models. Keep the checkpoint offline and collect
more independent build/late-wave recovery data before another retrain.

All recording, merge, training, and shadow commands used `HANDCRAFTED` or
`SHADOW_HUMAN`; no production controller code or hybrid deployment was changed.
