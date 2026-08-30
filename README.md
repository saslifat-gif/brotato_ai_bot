# Brotato AI

This repository contains one active V4 system for Brotato: a structured game
bridge, a single control pipeline, human-demonstration recording, training,
replay, and evaluation. The old screen-capture detector generations are no
longer part of the runtime.

## Setup

Windows 10/11 and Python 3.11 are supported. Install the requirements in the
`bota_ai` environment, then install the bridge and restart Brotato with
`--enable-mods`:

```powershell
conda activate bota_ai
Set-Location C:\ml\brotato
$env:PYTHONPATH = "src"
python -m v4.install_mod --game-dir "C:\Program Files (x86)\Steam\steamapps\common\Brotato"
```

The default policy is `HANDCRAFTED`. It is the existing production control
path. `SHADOW_HUMAN` only logs learned proposals; `HYBRID_HUMAN` remains off
unless it is explicitly approved after evaluation.

## Manual demonstrations

The recorder is observation-only: it never sends movement or menu actions.
Play manually and press F9 to bookmark a meaningful state. All inputs, state,
features, action transitions, builds, rewards, timestamps, and outcomes are
recorded automatically.

```powershell
$env:BROTATO_V4_POLICY_MODE = "HANDCRAFTED"
$env:BROTATO_V4_AUTOMATE_MENUS = "0"
python -m v4.record_human_demo `
  --output models\version_3\human_demos\manual_run01.sqlite `
  --run-label manual_run01 `
  --require-capture
```

Use `--continue-after-terminal` when recording repeated runs in one file.
Do not intentionally create deaths; mark naturally difficult states before
making the recovery action.

## Autonomous evaluation

Run the frozen baseline with the model and keep the learned human policy in
shadow mode when comparing it:

```powershell
$env:BROTATO_V4_POLICY_MODE = "SHADOW_HUMAN"
$env:BROTATO_V4_HUMAN_MODEL = "models\version_3\human_demos\human_event_bc.pt"
$env:BROTATO_V4_AUTOMATE_MENUS = "1"
python -m v4.run_frozen `
  --model models\version_3\human_base_ppo_recovery.zip `
  --policy model --episodes 3 `
  --results reports\shadow.json
```

## Common commands

```powershell
python -m v4.validate_human_demo <dataset.sqlite> --require-capture
python -m v4.report_human_demo_quality <run1.sqlite> <run2.sqlite> <run3.sqlite>
python -m v4.train_event_human_bc --dataset <dataset.sqlite> --report <report.json> --checkpoint <model.pt>
python -m v4.dagger_corrective --help
pytest
```

## Layout

```text
src/brotato_ai/     shared domain, bridge, control, data, policy, and evaluation code
v4/                 active runtime, trainer, recorder, evaluator, DAgger tools, and mod
tests/              unit, replay, and integration tests
docs/               architecture, operations, and dataset/evaluation notes
models/             local checkpoints and recordings (ignored by Git)
reports/            local evaluation output (ignored by Git)
```

The existing `models/version_3` folder name is retained only to avoid moving
user checkpoints; it is not a second code generation.

## License

MIT — see [`LICENSE`](LICENSE).
