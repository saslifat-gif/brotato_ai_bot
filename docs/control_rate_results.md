# Control-rate investigation results

Input: `models/version_3/raw_records/rate_experiment.jsonl`, one 60 Hz raw
recording with 70,472 rows. The same safety/controller pipeline was replayed
at each schedule. At every simulated tick it used the newest source frame
whose timestamp had arrived, then held that action until the next tick.

The passive recording cannot change its realized health, death, or projectile
hit labels under a counterfactual action. Those columns are therefore shown as
observed labels, while scheduling, action, escape, and geometry columns are
replay measurements.

## Rate sweep

| rate | decisions | actionable→next tick mean / p95 / max | mean stale source age | action-change Hz | oscillation Hz | hazard-window failure | observed HP-loss rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 11,828 | 53.94 / 97.00 / 99.00 ms | 11.99 ms | 0.754 | 0.091 | 9.30% | 0.0695% |
| 15 Hz | 17,741 | 35.72 / 64.33 / 65.67 ms | 11.80 ms | 0.830 | 0.101 | 9.30% | 0.0695% |
| 20 Hz | 23,655 | 28.30 / 48.00 / 49.00 ms | 11.94 ms | 0.892 | 0.115 | 9.30% | 0.0695% |
| 24 Hz | 28,386 | 20.58 / 38.33 / 40.33 ms | 10.60 ms | 0.936 | 0.117 | 9.30% | 0.0695% |
| 30 Hz | 35,482 | 19.20 / 31.33 / 32.33 ms | 11.82 ms | 0.944 | 0.121 | 9.30% | 0.0695% |
| 40 Hz | 47,310 | 12.74 / 23.00 / 24.00 ms | 10.60 ms | 0.974 | 0.133 | 9.30% | 0.0695% |
| 60 Hz | 70,964 | 10.51 / 15.00 / 16.33 ms | 11.79 ms | 1.043 | 0.133 | 9.30% | 0.0695% |

Observed deaths were zero in this recording and the observed projectile-hit
count was one at every rate, as expected for a passive replay. Mean/minimum
enemy separation and desired ranged-spacing fraction are also properties of
the recorded trajectory, not alternate physics outcomes: 417.24 / 5.21 units
and 16.80%, respectively.

| rate | post-escape re-entry | escape entries | escape reversals | safest-action rate | time in ESCAPE | desired ranged spacing | no-safe-action by next tick |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 34.04% | 94 | 55 | 67.76% | 5.54% | 16.80% | 0 |
| 15 Hz | 25.20% | 127 | 50 | 67.44% | 5.01% | 16.80% | 0 |
| 20 Hz | 19.50% | 159 | 47 | 67.31% | 4.70% | 16.80% | 0 |
| 24 Hz | 17.49% | 183 | 47 | 67.54% | 4.56% | 16.80% | 0 |
| 30 Hz | 15.45% | 220 | 48 | 67.59% | 4.34% | 16.80% | 0 |
| 40 Hz | 12.11% | 289 | 54 | 67.14% | 4.32% | 16.80% | 0 |
| 60 Hz | 8.47% | 413 | 50 | 67.64% | 4.07% | 16.80% | 0 |

The schedule delay decreases as expected, but the action-selection rate stays
near 0.67–0.68 and action churn rises only from about 0.75 to 1.04 changes per
second. The escape guard's eight-step hold is a known rate interaction: it is
333 ms at 24 Hz, 267 ms at 30 Hz, 200 ms at 40 Hz, and 133 ms at 60 Hz. This
is why escape time and re-entry metrics must not be interpreted as pure source
frequency effects without a wall-clock-normalized policy run.

## 15 Hz phase offsets

| phase offset | delay mean / p95 / max | post-escape re-entry | action-change Hz | oscillation Hz |
|---:|---:|---:|---:|---:|
| 0.0 ms | 35.72 / 64.33 / 65.67 ms | 25.20% | 0.982 | 0.119 |
| 16.667 ms | 36.43 / 64.00 / 66.33 ms | 26.15% | 1.004 | 0.145 |
| 33.333 ms | 36.00 / 64.00 / 65.67 ms | 25.98% | 1.017 | 0.141 |
| 50.0 ms | 33.89 / 64.00 / 65.33 ms | 25.20% | 0.992 | 0.128 |

The phase shift changes action metrics only slightly and does not provide a
survival comparison because the recording's physics is fixed. The maximum
15 Hz scheduling delay is 66.33 ms, matching the expected approximately
66.67 ms tick interval.

## Failure timing and counterfactual check

The failed-projectile TTI bucket contained one observed failure, in `>400 ms`;
the other TTI buckets were zero. At 24 Hz, 41 failed-hazard observations were
replayed with one additional policy decision inserted before impact. Zero of
41 selected an action that was materially safer at the impact frame; 29
selected the same action and 12 had no geometrically safe action available.

This recording therefore does not support the claim that an additional 20–50
ms decision would have prevented the observed failures. It supports a timing
claim—24 Hz imposes a roughly 40 ms maximum scheduling interval—but not a
causal gameplay-improvement claim. Paired live trials with the frozen same
checkpoint are still required to establish a material health/death benefit.

## Live-trial status

Bridge `0.3.20` is installed on the Windows game and accepts explicit 30/40/60
Hz requests while retaining a 24 Hz default. The live listener could not be
started during this pass because Windows currently excludes TCP ports
4182–4281, which includes the fixed bridge port 4242. No system port exclusion
was changed and no live health/death result is claimed here.
