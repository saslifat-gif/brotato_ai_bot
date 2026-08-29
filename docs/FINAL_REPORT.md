# Brotato AI — Human-Policy Integration: Final Report

Status: implementation complete, full remote suite green (Windows `bota_ai` env),
baseline evidence archived. Live A/B is a user-run handoff (see §9).

## 1. Mandate and decisions

Spec: `~/Downloads/Audit and Restructure the Bot Project for Human-Policy
Integration.md` (24 sections). Confirmed decisions:

| Decision | Outcome |
| --- | --- |
| Where the real project lives | Windows `lifat@192.168.1.3` → `C:\ml\brotato`; Mac copy is the working clone |
| Workflow | git round-trip: Windows → origin → Mac clone → bundle back to Windows |
| `HANDCRAFTED` mode semantics | exactly today's PPO + shield stack; unchanged default |
| WIP handling | existing WIP committed as one checkpoint first (commit `2f6b057`) |
| Protected file | `v1/shop/ocr_winmedia.py` is user-owned; never staged, edited, or reverted. Verified absent from every staged file set. |

## 2. Audit outcome

The v4 restructure (`feat/v4-restructure`, `docs/restructure/*`) already delivered
single-writer ownership (`FinalActionArbiter`/`FinalActionWriter`), normalized
`StateSnapshot`, unified hazard scoring, ms-based rate meters, and validated
config. The audit classified the remaining gaps:

**Must fix** — escape hold in control steps; no policy-mode system; no shadow /
hybrid gating; no runtime loader for the event model; no versioned checkpoint;
4 stale legacy tests in `test/v1/unit/test_v3_api.py`; stray duplicate
`v3/brotato_api_env.py`.

**Should fix** — `DecisionTrace` lacking learned-proposal fields; missing config
keys for model path / thresholds; no feature-parity guarantee between the demo
recorder and live inference; build-policy mode interface; unlabeled framewise-EC
scripts.

**Left alone** — arbiter/writer ownership, `DecisionTraceLogger`, `StateSnapshot`,
rate meters, v1/v2 packages, `docs/restructure/*` contracts.

## 3. What changed (branch `feat/human-policy-integration`)

New modules under `src/brotato_ai/policy/`:

- `modes.py` — `PolicyMode` (`HANDCRAFTED` / `SHADOW_HUMAN` / `HYBRID_HUMAN` /
  `EXPERIMENTAL_FULL_LEARNED`) + parsing; default `HANDCRAFTED`.
- `features.py` — `HumanPolicyFeatureBuilder`; reproduces the training input
  exactly (832-vector with slice 16:25 zeroed, rounded to 6 decimals, 0/200/400 ms
  trend differences, held-action one-hot → 2505-wide input). Parity with the
  recorder path is asserted by `tests/unit/test_human_feature_parity.py`.
- `human_action.py` — shared `EventHumanModel` (256→ReLU→Dropout→128, three
  heads), versioned checkpoint save/load (`brotato_event_human_bc` format)
  validated on load, and `EventHumanActionPolicy` adapter that never raises.
- `hybrid.py` — `DecisionTrigger` (ms interval + escape), `PersistenceManager`
  (real-time hold, default 438 ms), `HumanHybridController` (confidence-gated
  proposal → persistence → handcrafted fallback).

Modified:

- `v3/env/brotato_api_env.py` — optional learned stack built once at init; any
  load failure demotes to HANDCRAFTED with a warning; `_human_proposal` never
  raises; `_apply_human_policy` returns `(trace_fields, requested)` so shadow
  only records and hybrid replaces only **before** the arbiter.
- `domain/decisions.py` — `DecisionTrace` schema v2 (additive human_* fields;
  schema-1 keys unchanged; escape_remaining_ms added).
- `training/configs.py` — new env vars (`BROTATO_V4_POLICY_MODE`, `..._HUMAN_MODEL`,
  `..._HUMAN_CONFIDENCE`, `..._HUMAN_HOLD_MS`, `..._HUMAN_INTERVAL_MS`,
  `..._ALLOW_FULL_LEARNED`, `BROTATO_V4_BUILD_POLICY_MODE`); validation fails
  loudly on inconsistent combos; startup summary prints mode + model.
- `control/recovery.py` — escape holds now expressed in milliseconds
  (`hold_duration_ms`, `side_hold_duration_ms`; defaults byte-identical to the
  8-step / 6-side-step behavior at 24 Hz, verified by `test_escape_timing.py`).
- `control/arbiter.py` — forwards `control_interval_ms`; sets `escape_remaining_ms`.
- `ui/modes.py` (new) + `ui/build_policy.py` (facade) — build-policy mode
  contract; `LEARNED` refuses auto-discovered candidate checkpoints.
- `v3_event_human_bc.py` — `--checkpoint`; report JSON gains the checkpoint key;
  uses the shared model class.
- `v3/train_combat_bc.py`, `v3/train_human_demo_bc.py` — LEGACY labels; import-lint
  test proves nothing in `src/` or the active entrypoints imports them.
- Deleted: `v3/brotato_api_env.py` (stale duplicate; all importers use `v3/env/`).

## 4. Verification evidence

- **Local (Mac, python 3.13):** `tests/unit` green except 4 tests skipped (need
  gymnasium, available only in the Windows env).
- **Windows `bota_ai` full suite** (incl. legacy `test/`, the 6 sb3 PPO tests,
  and the 4 repaired legacy tests): **all green** — archived at
  `reports/human_policy_baseline/full_test_suite.log`.
- **Feature parity** — training-path vs live-builder input equality (atol 1e-5),
  writer→load_frames→build_examples vs live builder.
- **Timing semantics** — default hold identical to previous step behavior at
  24 Hz; duration stable across 12/48 Hz; step-count fallback preserved.
- **Shadow no-op** — SHADOW returns identical requested action and only records
  trace fields; hybrid replaces only pre-arbiter; inference failure survives.
- **Baseline backtest** (fixed recording, 4850 samples): archived at
  `reports/human_policy_baseline/backtest_report.{json,md}` — unified shield
  modeled-risk delta 0.268, escape re-entry reduced 61.2% → 0.0%, drift 0.

## 5. Learned-model contract recap

Checkpoint (format `brotato_event_human_bc`, schema v1): built by
`v3_event_human_bc.py --checkpoint`, loaded with `torch.load(weights_only=True)`,
validated for format/schema/feature-version/action-names, and used only through
`EventHumanActionPolicy` under the active mode. The duration head and the weak
change gate (F1 ~0.14) are diagnostics only — neither ever times or triggers
production transitions. Any failure degrades to HANDCRAFTED and logs; the
production loop cannot crash from the learned path.

## 6. Performance

No control-loop change on the HANDCRAFTED path (default): the learned stack is
suspended until a non-handcrafted mode is configured. In shadow/hybrid, the MLP
proposal is 256→128 (≪1 ms); feature extraction is measured before use. No live
timing was changed; `escape_remaining` (steps) is kept alongside
`escape_remaining_ms` for dashboard continuity.

## 7. Configuration surface (new)

`BROTATO_V4_POLICY_MODE` (default HANDCRAFTED) is the single mode switch;
shadow/hybrid require `BROTATO_V4_HUMAN_MODEL`; full-learned additionally requires
`BROTATO_V4_ALLOW_FULL_LEARNED=1`. `BROTATO_V4_BUILD_POLICY_MODE` governs build
selection (HANDCRAFTED default; LEARNED needs an explicit `BROTATO_V3_UI_MODEL`).

## 8. Remaining debt

1. Dataset diversity (varied builds, dense projectiles, recovery, boss waves):
   3 episodes / 454 MB is thin; collect with the existing recorder format.
2. Change-gate calibration toward trustworthy change probabilities before any
   timing handoff to the model.
3. A production checkpoint trained from `session_001.sqlite` (model artifact
   still to be produced; training happens in the `bota_ai` env).

## 9. Handoff (user-run on Windows)

1. `git pull` / fetch `feat/human-policy-integration` (this branch already
   merged here); push to origin for backup.
2. Train a real checkpoint:
   `set PYTHONPATH=src&& python v3_event_human_bc.py --dataset models\version_3\human_demos\session_001.sqlite --report reports\human_model.json --checkpoint models\version_3\human_event_bc.pt`
3. Live A/B against HANDCRAFTED (production entrypoint `v3/run_frozen.py`):
   ```
   set PYTHONPATH=src
   set BROTATO_V4_POLICY_MODE=SHADOW_HUMAN
   set BROTATO_V4_HUMAN_MODEL=C:\ml\brotato\models\version_3\human_event_bc.pt
   C:\Users\lifat\miniconda3\envs\bota_ai\python.exe v3\run_frozen.py --model <ppo.pt> --policy model --episodes 5 --results reports\shadow_ab.json
   ```
   `RuntimeConfig` is read from `BROTATO_V4_*` env vars via `v3/config.py`
   (the validated config surface), so no code change is needed to switch modes.
   Compare with a HANDCRAFTED run at the same episodes; watch the `human_*`
   DecisionTrace fields, override/disagreement rates, and damage/survival.
   Only after shadow logs look sane, switch to `HYBRID_HUMAN`.
4. Never ship `BROTATO_V4_POLICY_MODE` other than default or shadow without a
   review of shadow disagreement data.