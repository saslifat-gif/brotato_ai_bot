# Human gameplay demonstration pipeline

This pipeline is separate from the production controller. It records human
combat movement and human build/menu choices without sending movement or UI
commands. The source observation and analog input are retained; controller
metrics are analysis fields and can be regenerated.

## Capture on Windows

Install bridge `0.3.19`, restart Brotato with `--enable-mods`, and start the
recorder from `C:\ml\brotato` in the `bota_ai` environment:

```powershell
conda activate bota_ai
$env:PYTHONPATH = "src"
python -m v3.record_human_demo `
  --output models/version_3/human_demos/session_001.sqlite
```

Play normally. Combat and shop/upgrade choices are human-controlled. Stop with
Ctrl+C; the recorder finalizes the fixed-horizon outcomes and validation report.

The database contains:

- `frames`: rich state, processed/raw input, action label, synchronized
  monotonic timestamps, current safety/controller diagnostics, and features.
- `raw_samples`: the independent 60 Hz stream.
- `action_segments`: action start/end and persistence duration.
- `build_decisions`: available UI options and build snapshots.
- `transitions`: observed outcomes at 50/100/250/500/1000 ms.

If no joystick is connected, `raw_available` is false and the processed WASD
signal remains intact. The validator reports that limitation explicitly.

## Validate and inspect

```powershell
$env:PYTHONPATH = "src"
python -m v3.validate_human_demo models/version_3/human_demos/session_001.sqlite `
  --report models/version_3/human_demos/session_001.validation.json
python -m v3.replay_human_demo models/version_3/human_demos/session_001.sqlite --frame-id 1000
python -m v3.compare_human_controller models/version_3/human_demos/session_001.sqlite
```

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
