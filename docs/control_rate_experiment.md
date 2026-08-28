# Control-rate sensitivity experiment

Run the experiment against a 60 Hz raw recording with:

```text
PYTHONPATH=src python -m brotato_ai.evaluation.control_rate \
  models/version_3/raw_records/<recording>.jsonl \
  --json reports/control_rate.json
```

The harness evaluates 10, 15, 20, 24, 30, 40, and 60 Hz, plus 0, 16.667,
33.333, and 50 ms phase offsets at 15 Hz. It calls the existing v4
`CombatDecisionPipeline` unchanged, uses the newest observation available at a
scheduled tick, and holds the resulting action until the next tick. It also
reports action changes, opposite-direction oscillations, observation age at a
decision, and the configured step-based persistence context.

The JSON contains scheduling-delay statistics, hazard-window and TTI buckets,
rate/phase action metrics, and counterfactual failure categories. Since a
passive recording contains one realized game trajectory, health/death/hit
labels remain observed labels; they are not presented as alternate outcomes
under a different action schedule. A causal survival conclusion requires
paired live replays or a game-state simulator.

Before a full run, estimate wall time using a bounded sample:

```text
PYTHONPATH=src python -m brotato_ai.evaluation.control_rate \
  models/version_3/raw_records/<recording>.jsonl \
  --json reports/control_rate.json \
  --estimate-only --estimate-sample 200
```

The estimate benchmarks the same eleven-condition sweep used by the full run.

The live bridge remains at 24 Hz by default. The `0.3.20` bridge accepts an
explicit 30/40/60 Hz `configure` request so the frozen V4 runner can perform
paired live trials with the same checkpoint. No policy threshold, scorer,
prediction rule, or recovery rule is changed by selecting the rate. Because
the passive replay cannot re-simulate the game physics, its health/death/hit
columns are observed labels; use paired live runs for a causal gameplay claim.
