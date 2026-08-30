# Brotato AI V4 handoff

The repository has one active runtime. Shared contracts and control primitives
live under `src/brotato_ai`; game-facing runtime, recording, training,
evaluation, DAgger, and the bridge mod live under `v4/`.

## Safety rules

- `HANDCRAFTED` remains the default and is the production controller.
- `SHADOW_HUMAN` only observes and logs learned proposals.
- `HYBRID_HUMAN` stays disabled.
- Do not touch the user-owned `v1/shop/ocr_winmedia.py` file. It is retained
  only because it is explicitly protected and is not imported by V4.
- Existing checkpoints and recordings under `models/version_3/` are artifact
  compatibility paths, not a second code generation.

## Windows setup

```powershell
conda activate bota_ai
Set-Location C:\ml\brotato
$env:PYTHONPATH = "src;."
$env:BROTATO_V4_POLICY_MODE = "HANDCRAFTED"
$env:BROTATO_V4_AUTOMATE_MENUS = "0"
```

Install the bridge from the canonical V4 source, then restart Brotato:

```powershell
python -m v4.install_mod --game-dir "C:\Program Files (x86)\Steam\steamapps\common\Brotato"
```

## Manual recording

`v4.record_human_demo` is observation-only. The human plays combat and every
menu/build choice; Python sends no movement or menu action. The recorder stores
raw input, structured state, derived features, reward components, transitions,
build choices, timestamps, episode boundaries, and outcomes. F9 creates a
repeatable observation bookmark.

```powershell
python -m v4.record_human_demo `
  --output models\version_3\human_demos\manual_run01.sqlite `
  --run-label manual_run01 `
  --continue-after-terminal `
  --require-capture
```

Do not intentionally die to create labels. Mark difficult states naturally,
then make the recovery decision yourself.

## Offline event-policy workflow

```powershell
python -m v4.validate_human_demo <dataset.sqlite> --require-capture
python -m v4.report_human_demo_quality <run1.sqlite> <run2.sqlite> <run3.sqlite>
python -m v4.train_event_human_bc `
  --dataset <dataset.sqlite> `
  --report reports\human_event_bc.json `
  --checkpoint models\version_3\human_event_bc.pt
```

Keep corrective labels human-authored. Do not convert model proposals,
handcrafted actions, safety overrides, or counterfactual risk scores into
synthetic labels.

## Shadow evaluation

```powershell
$env:BROTATO_V4_POLICY_MODE = "SHADOW_HUMAN"
$env:BROTATO_V4_HUMAN_MODEL = "models\version_3\human_event_bc.pt"
$env:BROTATO_V4_AUTOMATE_MENUS = "1"
python -m v4.run_frozen `
  --model models\version_3\human_base_ppo_recovery.zip `
  --policy model --episodes 3 `
  --results reports\shadow.json
```

The handcrafted action remains applied. Review agreement, confidence,
calibration, safety/risk diagnostics, wave/build coverage, and manually inspect
high-confidence disagreements before considering any policy change.

## Validation status of this restructure

- V1/V2/V3 implementation trees and obsolete launchers were removed.
- The active API runtime was absorbed into `v4`; no `v1`, `v2`, or `v3` Python
  imports remain in `src`, `v4`, or `tests`.
- The protected OCR file was not modified.
- `compileall`, dynamic imports, and all V4 CLI help checks pass locally.
- Run the full suite in the Windows `bota_ai` environment before syncing the
  branch to the game machine.

The canonical documentation is `README.md`, `docs/ARCHITECTURE.md`, and the
focused contracts under `docs/restructure/`.
