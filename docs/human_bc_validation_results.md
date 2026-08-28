# Human BC validation results

The exact checkpoint associated with the reported 93.1% result was evaluated
without changing the production controller:

- Dataset: `models/version_3/human_demos/session_001.sqlite`
- Checkpoint: `models/version_3/human_demos/session_001_human_bc.pt`
- 41,894 feature rows, including 25,535 combat rows
- 3 complete episodes
- Original seeded split reproduced exactly: 34,535 training rows from 2
  episodes and 7,359 validation rows from 1 complete episode (4,503 combat
  rows)
- The feature width is 832 and the training script performs no normalization
  or dataset-wide statistic fitting

## Main result

| Model | All frames | Non-transition | Transition |
|---|---:|---:|---:|
| Majority training action | 14.31% | 15.06% | 4.00% |
| Previous-action persistence | **93.21%** | 100.00% | 0.00% |
| BC checkpoint | **93.14%** | **99.93%** | **0.00%** |

On the held-out episode, persistence is slightly better than BC by 0.07
percentage points. BC balanced accuracy is 87.78% and macro F1 is 87.77%,
versus 87.86% and 87.86% for persistence. The majority baseline is only
14.31% accurate, so this is not simple majority-class collapse; it is almost
entirely action persistence.

There were 500 real human action transitions. BC predicted the new action on
0 of the 500 exact transition frames. Its top-3 transition accuracy was
93.6%, and the new action appeared somewhere within ±1 recorded frame for
99.8% of transitions. This means the model often recognizes the new direction
after the transition, but does not predict the decision at the decision
timestamp.

The change detector confirms this:

- predicted-change precision/recall/F1: 0.0% / 0.0% / 0.0%
- ROC-AUC from the model's probability of changing: 0.657
- average precision: 0.148
- exact new-action accuracy conditioned on a real change: 0.0%

## Previous-action ablation

The semantic vectorizer exposes previous action as the one-hot feature indices
16–24.

| Variant | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Full checkpoint | 93.14% | 87.78% | 87.77% |
| Full checkpoint, previous-action feature zeroed at inference | 27.49% | 31.86% | 25.36% |
| Retrained diagnostic BC without previous-action feature | 41.05% | 42.60% | 41.37% |

The ablation shows that the reported accuracy depends heavily on the explicit
previous-action feature. The no-previous-action model still learns some state
signal, but it is not close to the full model's frame accuracy.

## Hold behavior and autoregressive rollout

Using the same frame timing estimate, human holds on the held-out episode had
mean duration 470 ms, median 160 ms, and p90 544 ms. Teacher-forced BC had
mean 464 ms, median 160 ms, and p90 544 ms, with 2.15 action changes per
second versus 2.12 for the human. It also produced zero rapid reversals,
which looks realistic only because it follows the recorded previous action.

In an autoregressive offline rollout, the model's own previous prediction was
fed back into the previous-action feature:

- teacher-forced accuracy: 93.14%
- autoregressive accuracy: 11.90%
- agreement with teacher-forced predictions: 11.99%
- first divergence: frame offset 289, approximately 9.25 seconds into the
  held-out episode
- autoregressive action changes: 0.051 per second, with only 13 predicted
  segments across 7,359 frames

This is the strongest evidence that the 93.1% frame score is not a stable
live-policy score. The model becomes self-consistent with its own stale action
and stops tracking the human sequence.

## Held-out episode generalization

The original checkpoint's held-out episode result is the primary result. Since
there are only 3 episodes, a diagnostic leave-one-episode-out retraining study
was also run:

| Held-out episode | Frames | BC accuracy | Persistence | BC transition accuracy |
|---|---:|---:|---:|---:|
| Non-combat episode | 2,957 | 99.97% | 99.97% | 0.0% |
| Large combat episode | 31,578 | 58.48% | **93.89%** | 5.13% |
| Original validation episode | 7,359 | 93.14% | **93.21%** | 0.0% |

The large combat fold is particularly revealing: after retraining without that
episode, BC falls to 58.5% while persistence remains 93.9%. Broad generalization
to new episodes, builds, or runs is therefore not demonstrated.

## Context and build checks

On the primary held-out episode, BC did not beat persistence in any of the
main tactical buckets:

- actionable hazard: BC 92.82%, persistence 92.89%
- inside desired ranged spacing: BC 91.19%, persistence 91.19%
- high HP: BC 90.26%, persistence 90.30%
- low HP: BC 99.82%, persistence 99.95%
- nearest enemy under 100 units: BC 85.27%, persistence 87.05%
- 1–4 projectiles: BC 90.24%, persistence 90.28%

The recording contains several build snapshots, including SMG states and a
full item/stat snapshot, but they are from one gameplay session. Independent
build-conditioned generalization has not been established.

## BC versus safety recommendation

On the held-out episode, the three-way disagreement counts were:

| Category | Frames |
|---|---:|
| BC = human, safety differs | 5,593 |
| Safety = human, BC differs | 67 |
| BC = safety != human | 48 |
| All three differ | 390 |
| All three agree | 1,261 |

Representative cases are retained in the machine-readable report. Most
BC=human/safety-differs frames are continuity frames where the human holds an
action while the safety scorer prefers another lane; most have no observed
short-horizon damage. The stored 250/1000 ms outcomes belong to the realized
human action and cannot prove that the BC or safety alternative would have
performed better without a physics counterfactual.

## Conclusion

The 93.1% figure is not evidence that the model has learned human decision
timing. It is a strong frame-reconstruction score dominated by temporal
leakage through `previous_action`, with the model effectively predicting “keep
doing what was already being done.” It does not detect action changes at the
correct frame, and its autoregressive behavior collapses after its first
mistake.

The model is therefore suitable as a diagnostic learning baseline, but not as
evidence of a reliable learned movement policy. The next useful experiment is
to train/evaluate a decision-event or short-horizon target with strict
episode-level holdouts, while leaving the current production controller
unchanged.
