# Human gameplay demonstration pipeline

This pipeline is separate from the production controller. It records human
combat movement and human build/menu choices without sending movement or UI
commands. The source observation and analog input are retained; controller
metrics are analysis fields and can be regenerated.

## Capture on Windows

Install bridge `0.3.22` from this branch, restart Brotato with
`--enable-mods`, and start the recorder from `C:\ml\brotato` in the
`bota_ai` environment. The recorder is observation-only: it never sends
movement or UI actions. Keep the production policy explicitly handcrafted and
keep menus manual for this collection:

```powershell
conda activate bota_ai
$env:PYTHONPATH = "src"
$env:BROTATO_V4_POLICY_MODE = "HANDCRAFTED"
$env:BROTATO_V3_AUTOMATE_MENUS = "0"
python -m v3.record_human_demo `
  --output models/version_3/human_demos/manual_run01.sqlite `
  --run-label manual_run01 `
  --notes "non-SMG build; prioritize waves 8-12, recoveries, dense threats, and low HP" `
  --require-capture
```

Repeat for `manual_run02.sqlite` and `manual_run03.sqlite`, choosing different
non-SMG or mixed builds where the offers allow it. If the recorder starts on a
previous run's death/victory screen, it waits for the next non-terminal frame
and does not count that stale screen as a new episode. It then stops
automatically on the first terminal frame of the accepted run; if you must
abort before then, use Ctrl+C and treat that file as incomplete. Use the same
bridge and recorder settings for all three runs. Do not set `HYBRID_HUMAN`, start
`run_frozen.py`, or launch a training process during collection.

For targeted repeated-death/recovery review, add
`--continue-after-terminal` to the recorder command and press `F9` in the game
whenever you want to bookmark a state. Each press is stored separately with an
exact timestamp and full state snapshot; F9 does not change movement or consume
the game controller input.

The database contains:

- `frames`: rich state, processed/raw input, action label, synchronized
  monotonic timestamps, safety diagnostics, derived tactical features, and
  shared API reward components.
- `raw_samples`: the independent 60 Hz stream, including raw stick/buttons and
  the rich-episode association when it is available.
- `action_segments`: action start/end and full persistence duration, rather
  than only the final frame interval.
- `build_decisions`: every observed UI snapshot, including all advertised
  choices, before/after build state, and selected/inferred result metadata.
- `transitions`: observed outcomes at 50/100/250/500/1000 ms.

The recorder stores the action-independent `ApiRewardEngine` components. It
marks them as observational because the live environment also has
controller-dependent movement shaping; no reward is used to alter human input.

If no joystick is connected, `raw_available` is false and the processed WASD
signal remains intact. The validator reports that limitation explicitly.

## Validate and inspect

```powershell
$env:PYTHONPATH = "src"
python -m v3.validate_human_demo models/version_3/human_demos/session_001.sqlite `
  --report models/version_3/human_demos/session_001.validation.json `
  --require-capture
python -m v3.replay_human_demo models/version_3/human_demos/session_001.sqlite --frame-id 1000
python -m v3.compare_human_controller models/version_3/human_demos/session_001.sqlite
```

After all three runs, produce the set-level report:

```powershell
python -m v3.report_human_demo_quality `
  models/version_3/human_demos/manual_run01.sqlite `
  models/version_3/human_demos/manual_run02.sqlite `
  models/version_3/human_demos/manual_run03.sqlite `
  --output models/version_3/human_demos/manual_set_quality.json `
  --markdown models/version_3/human_demos/manual_set_quality.md
```

The set report checks combat-frame and genuine-transition counts, build and
wave coverage, low-health/dense-threat/recovery coverage, transition entropy,
available-choice capture, SQLite/blob integrity, source-clock drift, and
training/live event-feature parity. It returns a nonzero status when the
recordings are not yet diverse enough for retraining.

The `safest_action` comparison is the unchanged shared hazard/recovery
architecture evaluated offline. It is not mislabeled as a reconstruction of a
trained policy that was not present in the recording.

## BC baseline

The baseline uses grouped episode holdout, so frames from one episode cannot
leak into both train and validation sets:

```powershell
$env:PYTHONPATH = "src"
python -m v3.train_human_demo_bc `
  models/version_3/human_demos/session_001.sqlite `
  --output models/version_3/human_demos/human_bc.pt
```

The semantic feature vector is only a baseline input. Future analog BC should
train against `input_blob.processed_stick` and, when available,
`input_blob.raw_stick`, while retaining the discrete action as an auxiliary
target.

## Dataset limitations

The bridge's production rich-state channel defaults to 24 Hz to preserve
control latency. The independent raw channel is 60 Hz, and the bridge accepts
up to 60 Hz for controlled rate experiments. Therefore the dataset has precise
60 Hz input/kinematics samples plus rich semantic snapshots at the configured
state rate; it does not invent semantic values between snapshots.
This avoids treating queued or stale frames as observations. A future bridge
optimization can raise rich-state capture separately if profiling shows it is
safe, without changing the dataset schema.
