# Brotato AI — Architecture and Human-Policy Integration

This document is the current architecture map required by the
human-policy integration restructure.  It extends, and does not replace, the
contracts in `docs/restructure/` (BASELINE, OWNERSHIP, FEATURE_GATES,
TACTICAL_MOVEMENT, TELEMETRY).

## Before (audit summary)

The v4 restructure had already established one action-resolution pipeline
(`FinalActionArbiter` + `FinalActionWriter`), a normalized `StateSnapshot`,
async bounded decision logging, and validated configuration.  The audit found
these gaps for human-policy integration:

| Finding | Class | Resolution |
| --- | --- | --- |
| No runtime loader/adapter for the event human model (offline-only, no checkpoint file) | must fix | `brotato_ai/policy/human_action.py` + `--checkpoint` in `v3_event_human_bc.py` |
| No policy-mode system (nothing between "current behavior" and "learned") | must fix | `brotato_ai/policy/modes.py` + `RuntimeConfig.policy_mode` |
| No shadow mode, no hybrid gating | must fix | env `_apply_human_policy` seam + `brotato_ai/policy/hybrid.py` |
| Escape hold expressed in control steps (`hold_steps=8`) | must fix | duration-based hold in `TacticalMovementController`, behavior-identical at 24 Hz |
| Four stale legacy tests in `test/v1/unit/test_v3_api.py` | must fix | repaired to assert current intended behavior |
| Event model architecture defined only inside the offline training function | must fix | shared `EventHumanModel` in `brotato_ai.policy.human_action` |
| Stray duplicate `v3/brotato_api_env.py` (stale copy of `v3/env/brotato_api_env.py`) | must fix | removed; `v3/env/brotato_api_env.py` remains the owner |
| `DecisionTrace` had no learned-proposal fields | should fix | schema v2 additive fields |
| No feature-parity guarantee between demo features and live inference | should fix | `HumanPolicyFeatureBuilder` + parity test against the recorder path |
| Build-policy learned candidates auto-discovered from the output dir | should fix | `BuildPolicyMode` gate; `LEARNED` requires an explicit model path |
| Framewise BC scripts unlabeled, reachable | should fix | LEGACY labels + import-lint test |
| v1/v2 packages, `docs/restructure/*` contracts, ownership map | leave alone | untouched |

## After (module boundaries)

```text
src/brotato_ai/
  domain/        actions, StateSnapshot, DecisionTrace (schema 2)
  bridge/        protocol, client, measured rates
  control/       UnifiedHazardScorer, TacticalMovementController (ms holds),
                 FinalActionArbiter (single override authority), FinalActionWriter
  policy/        NEW - learned human-policy integration layer
    modes.py         PolicyMode enum + parsing
    features.py      HumanPolicyFeatureBuilder (training-parity inputs)
    human_action.py  EventHumanModel, checkpoint save/load, fail-safe adapter
    hybrid.py        DecisionTrigger, PersistenceManager (ms), HumanHybridController
  data/          demo recorder, schema, replay, bounded cache
  training/      validated RuntimeConfig, callbacks
  evaluation/    backtest, control-rate experiments, metrics, reports
  ui/            build-policy mode contract, menu controller
v3/              active runtime entrypoints (env, combat policy, mod, recorders)
v4/              temporal hierarchical trainer and frozen runner
tests/           unit / replay / integration
```

Import ownership is unchanged from `docs/restructure/OWNERSHIP.md` with one
documented exception: `brotato_ai/policy/features.py` may lazily import
`v3.combat_policy.SemanticCombatVectorizer`, because that class *is* the
training feature definition and duplicating it would guarantee drift.

## Runtime paths

```text
HANDCRAFTED (default, byte-identical production):
    PPO policy request
      -> FinalActionArbiter (unified hazards -> tactical escape)
      -> FinalActionWriter

SHADOW_HUMAN:
    handcrafted path acts exactly as HANDCRAFTED
    human policy predicts silently each step
    proposal + confidence + agreement logged in the DecisionTrace only

HYBRID_HUMAN:
    DecisionTrigger (tactical escape state OR decision_interval_ms elapsed)
      -> human action head proposal (confidence-gated)
      -> PersistenceManager holds the chosen action in real time
      -> FinalActionArbiter (safety may still override)
      -> FinalActionWriter

EXPERIMENTAL_FULL_LEARNED (double-gated; never a default):
    human proposal every step; persistence bypassed
    requires BROTATO_V4_POLICY_MODE=EXPERIMENTAL_FULL_LEARNED
             and BROTATO_V4_ALLOW_FULL_LEARNED=1
    the FinalActionArbiter still vetoes catastrophic proposals
```

Mode selection: `BROTATO_V4_POLICY_MODE`; misconfiguration fails validation
loudly (e.g. shadow/hybrid without `BROTATO_V4_HUMAN_MODEL`).

## Learned-model contract

- **Checkpoint**: produced by `python v3_event_human_bc.py --dataset ... --report ... --checkpoint <path>.pt`.
  Payload: format `brotato_event_human_bc`, checkpoint schema 1, feature
  schema version, action names, state/input widths, history offsets,
  previous-action slice, max hold, model state dict, normalization mean/std
  (state part only), calibrated change threshold, held-out metrics, dataset
  name, seed.  Loads with `torch.load(..., weights_only=True)`.
- **Input**: `HumanPolicyFeatureBuilder.build_input(held_action)` — semantic
  state vector (832) with slice 16:25 zeroed, rounded to 6 decimals exactly
  as the recorder stores it; plus 0/200/400 ms trend differences; plus the
  held-action one-hot.  Parity with training is asserted in
  `tests/unit/test_human_feature_parity.py` through the real recorder path.
- **Output**: `HumanProposal(action, probability, probabilities,
  change_probability, duration_ms, held_action)`; selected action is the
  argmax excluding the held action.
- **Timing**: decision intervals and persistence are milliseconds
  (`BROTATO_V4_HUMAN_INTERVAL_MS`, `BROTATO_V4_HUMAN_HOLD_MS`; both default
  438 ms, the observed human hold mean).  The learned duration head is a
  diagnostic only (MAE 165 ms median; uncalibrated) and is never used to
  time production actions.
- **Change gate**: experimental.  Held-out change F1 ~0.14 (docs/
  event_human_imitation_results.md).  Available for offline evaluation and
  shadow logging; it never triggers or times production transitions.
- **Fallback**: any load failure demotes the mode to HANDCRAFTED with a
  printed warning; any inference failure returns `None` for that step and
  increments a counter.  The production loop cannot crash from the learned
  path, and `HYBRID_HUMAN`/`SHADOW_HUMAN` without a working model behave as
  HANDCRAFTED.

## Build-policy contract

```text
current build + available options + stage context + player stats
  -> build policy -> selected build option
```

- `HANDCRAFTED` (default): `RangedSmgTeacher` / `StickMeleeTeacher` rules,
  exactly as production today.
- `HUMAN_RECORDED`: human choices continue to be captured in the decision
  log; selection remains handcrafted.
- `LEARNED`: model refines teacher-gated ranking; enabled only with an
  explicitly configured `BROTATO_V3_UI_MODEL` (auto-discovered candidates
  are refused).  Build data (available alternatives, human choice,
  resulting stats) keeps being recorded for training.

Movement policy conditioning on build context (weapon range, weapon counts,
move speed, armor, attack speed) already enters both the v4 observation and
the semantic human-policy features at indices 25:32
(`v3/combat_policy.py`), so melee and ranged builds are represented; no new
feature is introduced by this restructure.

## Configuration map

`brotato_ai/training/configs.py` remains the single validated configuration
surface.  New keys: `BROTATO_V4_POLICY_MODE`, `BROTATO_V4_HUMAN_MODEL`,
`BROTATO_V4_HUMAN_CONFIDENCE`, `BROTATO_V4_HUMAN_HOLD_MS`,
`BROTATO_V4_HUMAN_INTERVAL_MS`, `BROTATO_V4_ALLOW_FULL_LEARNED`,
`BROTATO_V4_BUILD_POLICY_MODE`.  The startup summary prints the active
policy mode, build policy mode, and human model path.

## Remaining debt / next steps

1. Collect diverse demonstrations (varied builds, dense projectiles,
   low-health recoveries, boss waves) — the dataset, not the model, is the
   bottleneck; merge sessions with the existing recorder format.
2. Iterate the change gate toward calibrated change probabilities before
   ever moving transition timing to the model.
3. Live A/B for `SHADOW_HUMAN` then `HYBRID_HUMAN` against the HANDCRAFTED
   baseline using the FEATURE_GATES checklist (fixed-recording comparison,
   damage/survival/victory metrics, override and disagreement rates).
4. `TacticalMovementController` escape telemetry keeps the step-based
   `escape_remaining` field for continuity; dashboards should migrate to
   `escape_remaining_ms`.
