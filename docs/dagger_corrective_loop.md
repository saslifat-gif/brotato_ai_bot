# DAgger corrective loop for the human policy

This workflow is offline-first and keeps the production controller unchanged.
The live collector must run with `SHADOW_HUMAN`; the applied action remains the
handcrafted/PPO baseline action. `HYBRID_HUMAN` is not enabled by any command
below.

For repeated in-game death/recovery bookmarks, use the human recorder in
`HANDCRAFTED` mode and press `F9` whenever the state should be revisited. Each
press creates a separate marker with its own timestamp and full state snapshot;
`--continue-after-terminal` keeps the recorder alive across repeated runs and
terminal screens. The key is observation-only and does not alter movement or
production control:

```text
python v3\record_human_demo.py --output models\version_3\human_demos\marked_runs.sqlite --run-label marked-runs --continue-after-terminal --require-capture
```

The recorder stores these as `manual_bookmark` labels in SQLite. Press F9 once
per event you want to review; pressing it repeatedly is allowed and does not
collapse distinct marks. The bridge prints the marker id, phase, and wave so
the operator can confirm that the key was seen.

## Capture

The existing Windows `shadow_eval.py` accepts `--capture-full-state` and
`--full-state-log`. The decision JSONL contains compact references; the
sidecar stores each exact pre-action state once. This avoids duplicating long
histories in every decision row. A selected queue row later contains the full
state and the two historical samples used by the 0/200/400 ms event feature
contract.

Example Windows setup:

```text
set BROTATO_V4_POLICY_MODE=SHADOW_HUMAN
set BROTATO_V4_HUMAN_MODEL=C:\ml\brotato\models\version_3\human_demos\human_event_bc_manual_set.pt
set BROTATO_V4_BUILD_POLICY_MODE=HANDCRAFTED
set BROTATO_V3_AUTOMATE_MENUS=1
python shadow_eval.py --model models\version_3\human_base_ppo_recovery.zip --baseline-kind ppo --dataset models\version_3\human_demos\event_training_manual_set.sqlite --episodes 3 --output reports\shadow_dagger_decisions.jsonl --full-state-log reports\shadow_dagger_states.jsonl --trace-log reports\shadow_dagger_trace.jsonl --results reports\shadow_dagger_results.json --analysis reports\shadow_dagger_analysis.json --capture-full-state
```

For a machine with the observed evaluator memory growth, use bounded batches
(`--max-steps 300`) and start a fresh process for each batch. A bounded batch
is valid diagnostic data but must be marked partial when interpreting wave
coverage.

## Select and validate

```text
python v3\dagger_corrective.py select --shadow-log reports\shadow_dagger_decisions.jsonl --state-log reports\shadow_dagger_states.jsonl --human-dataset models\version_3\human_demos\event_training_manual_set.sqlite --output reports\dagger_queue.jsonl --budget 200 --min-gap-ms 500 --holdout-fraction 0.25 --report reports\dagger_queue.report.json
python v3\dagger_corrective.py validate --queue reports\dagger_queue.jsonl --report reports\dagger_queue.validation.json
python v3\dagger_corrective.py label --queue reports\dagger_queue.jsonl --database models\version_3\human_demos\dagger_corrections.sqlite --init-only
python v3\dagger_corrective.py review --queue reports\dagger_queue.jsonl --output reports\dagger_queue.review.html
```

Selection is deterministic and prioritizes disagreement, high confidence,
counterfactual safety override, hard/much-higher modeled risk, low HP, dense
combat, bad positioning, and representation-level OOD. The 500 ms gap avoids
turning a long hold into hundreds of duplicate labels while retaining the
original timestamps. Summary-only shadow logs are rejected.

Open the generated `review.html` in a browser before labeling. It renders a
tactical map from the exact captured player, enemy, projectile, indicator, and
pickup coordinates, and shows the current, handcrafted, learned, confidence,
risk, and safety context. It is an offline reconstruction rather than a pixel
screenshot; use game replay for ambiguous high-risk cases. The page exports a
human-authored JSONL that can be passed to `label --labels-jsonl`.

The final `label` command is either interactive or consumes a human-authored
JSONL. The latter must contain only real labels, for example:

```json
{"queue_id":"...","human_corrective_action":"UP_LEFT","hold_duration_ms":420}
```

Skipped and unlabeled rows remain in the SQLite audit table and never become
training examples.

## Merge, retrain, and evaluate

After manual labeling, merge only the corrective training split. The holdout
is excluded by default:

```text
python v3\dagger_corrective.py merge --base-dataset models\version_3\human_demos\event_training_manual_set.sqlite --corrections models\version_3\human_demos\dagger_corrections.sqlite --output models\version_3\human_demos\event_training_manual_set_dagger.sqlite
set PYTHONPATH=src
python v3_event_human_bc.py --dataset models\version_3\human_demos\event_training_manual_set_dagger.sqlite --report reports\human_manual_set_dagger_event_bc.json --framewise-report models\version_3\human_demos\session_001_bc_diagnostics.json --seed 7 --epochs 20 --negative-ratio 4 --checkpoint models\version_3\human_demos\human_event_bc_manual_set_dagger.pt
python v3\dagger_corrective.py evaluate --corrections models\version_3\human_demos\dagger_corrections.sqlite --checkpoint models\version_3\human_demos\human_event_bc_manual_set_dagger.pt --human-dataset models\version_3\human_demos\event_training_manual_set.sqlite --report reports\dagger_corrective_holdout.json
```

The corrective holdout reports action accuracy, change F1, calibration, model
risk relative to the handcrafted action, counterfactual safety overrides, and
feature distribution shift. Autoregressive accuracy is intentionally not
computed by chaining independent intervention states; doing so would invent
temporal continuity. Complete held-out episodes remain the primary event-model
evaluation, and the exact same repaired-PPO `SHADOW_HUMAN` comparison must be
rerun afterward.

## First validation batch

The first bounded validation artifact was one partial 300-step shadow batch:

- 300 full-state sidecar rows and 300 decision rows;
- 33 selected queue rows after the 500 ms deduplication gap;
- 24 train / 9 corrective holdout rows;
- all selected rows have full state, two historical samples, monotonic source
  timestamps, and 832-wide semantic features;
- this batch is one wave-10 `lightning_shiv` build, so it is not evidence of
  multi-build or late-wave generalization;
- compared with 64,478 normal human combat feature rows, the bot-state
  representation showed standardized mean gap 0.248, mean quantile gap 0.297,
  and mean per-state RMS z-score 0.763. These are distribution diagnostics,
  not causal safety measurements.

No corrective action labels have been added yet, no policy has been retrained
from this queue, and no checkpoint has been deployed. The queue is therefore
ready for manual labeling but not yet training-ready.
