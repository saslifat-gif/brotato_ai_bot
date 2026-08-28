# Event-based human imitation baseline

This is a separate offline experiment. It does not modify the production
controller, bridge, or the earlier framewise BC checkpoint.

## Design

Input is the current action plus a structured 400 ms state-history window:

- current state
- current-minus-200 ms state trend
- current-minus-400 ms state trend
- current action one-hot

The old previous-action one-hot is removed from the 832-value state vector
before temporal features are constructed. The current action is then supplied
explicitly to answer the intended question: hold it or change it.

The multi-head diagnostic model predicts:

1. hold versus change;
2. the next movement action, trained only on actual changes;
3. remaining hold time, as an auxiliary regression target.

Training uses all 1,930 observed transition events from the two training
episodes plus 7,720 weighted hard-negative hold frames. Hard negatives are
holds near an upcoming transition, hazards, projectile pressure, close enemies,
or combat pressure. State normalization is fitted from those training examples
only.

## Primary complete-episode holdout

The split exactly follows the original seeded episode split: 34,535 training
rows from two episodes and 7,359 rows from one complete held-out episode.
There are no temporally adjacent rows across the split.

| Measure | Old framewise BC | Event BC |
|---|---:|---:|
| Frame/action accuracy, teacher forced | 93.14% | not the primary target |
| Exact transition/change F1 | 0.00 | 0.140 |
| Next-action accuracy at true changes | 0.00% | **80.2%** |
| Transition timing mean absolute error | not meaningful | 124 ms |
| Autoregressive action accuracy | 11.9% | **35.2%** |
| Autoregressive change F1 | 0.00 | 0.088 |

The event model detected 59 of 500 true changes on the held-out episode, with
283 false positive changes. Thus it has real next-action signal once a change
is identified, but the hold/change gate is still weak: precision 17.3%, recall
11.8%, F1 14.0%.

Predicted hold duration is not yet calibrated: median absolute error is 165 ms
and mean absolute error is 472 ms. It should remain an auxiliary diagnostic,
not a live action-duration controller.

## Context

On the primary holdout, the model selected the correct new action for 80.2% of
real transitions. That figure remains similar across tactical contexts, for
example 81.1% with 1–4 projectiles, 79.3% with an enemy inside 100 units, and
80.4% at high HP. In contrast, change F1 is low in every bucket; detecting
when to terminate the action is the limiting problem, not choosing a direction
after a true transition has been supplied.

## Complete-episode generalization

Diagnostic leave-one-episode-out training confirms that this is a useful
baseline rather than an integration candidate:

| Held-out episode | Frames | Change F1 | Next-action accuracy | Timing MAE | AR action accuracy |
|---|---:|---:|---:|---:|---:|
| Non-combat episode | 2,957 | 0.000 | 100.0% on one event | n/a | 0.0% |
| Large combat episode | 31,578 | 0.147 | 35.7% | 75 ms | 19.4% |
| Original held-out combat episode | 7,359 | 0.140 | 80.2% | 124 ms | 35.2% |

The model improves materially on the framewise baseline's transition behavior,
but its generalization remains too inconsistent for deployment. The large
combat episode contains substantially different action and state dynamics.

## Conclusion

The redesign validates the target change: transition supervision exposes real
state-to-next-action signal that framewise BC hid behind persistence. However,
the current data is too narrow for robust termination timing and autoregressive
control. Keep this as an offline diagnostic baseline.

The next data collection should add independent gameplay sessions with varied
builds, dense projectiles, low-health recoveries, boss waves, and deliberate
action changes. The next model iteration should focus on calibrated change
probabilities and transition-time labels rather than increasing model size.
