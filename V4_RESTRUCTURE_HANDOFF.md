# Brotato AI Bot — v4 Restructure Handoff

## Purpose

This document is the working contract for restructuring the active v4 project
into a cleaner, more practical, and more reviewable system. It records the
current behavior that has evidence behind it, the architecture that should be
preserved, and the gates that every future feature must pass before it is
enabled in live training.

The project should be treated as one active v4 system. Do not add new behavior
to disconnected historical packages or create another parallel controller.

## Current operating context

- Windows host: `lifat@192.168.1.3`
- Repository: `C:\ml\brotato`
- Branch: `feat/v3-api-agent`
- Training environment: `C:\Users\lifat\miniconda3\envs\bota_ai`
- Action bridge: port `4242`
- Raw recorder: port `4243`
- Active TensorBoard dashboard: `http://127.0.0.1:6007/`
- Raw recording target: approximately 60 Hz
- Current live control stream observed during testing: approximately 14.65 Hz
- Intended live control target: 24 Hz
- Asset-library limit: 10 GB maximum

The checkout is currently ahead of its remote branch and contains uncommitted
work. Create a clean snapshot or branch before beginning the large restructure.
The unrelated change in `C:\ml\brotato\v1\shop\ocr_winmedia.py` belongs to the
user and must not be overwritten, reverted, or folded into the restructure.

## Current implementation status

The following active behavior is already implemented and should be preserved
unless a replay experiment shows a measurable regression:

- The game bridge and trainer are separate processes.
- Raw recording is independent from the lower-rate control loop.
- The temporal actor receives multi-frame movement and threat history.
- Enemy attacks have semantic categories such as contact, charge, summon,
  projectile, and area.
- Boss-owned attacks and telegraphs are linked to their runtime enemy owner.
- The safety calculation predicts enemy movement using position and velocity,
  and also consumes bridge-provided enemy, projectile, and boundary path risk.
- Boss bodies, boss projectiles, and boss telegraphs receive higher danger
  weighting.
- Training telemetry distinguishes measured control rate from training
  throughput.
- PPO training can use CUDA, but the live decision path is intentionally a
  lightweight CPU calculation.

The latest source refactor is installed in the Windows worktree, but a running
Python process keeps the code that it imported at startup. A normal trainer
restart is required before source changes become live.

## Evidence from the controller backtest

The same raw recording was replayed with four controller structures plus a
stable version of the unified structure. These are geometric counterfactuals:
they measure predicted hazard risk, not guaranteed alternate game outcomes.

| Structure | Mean modeled risk | Selected minimum-risk action | Override rate |
| --- | ---: | ---: | ---: |
| Recorded policy | 0.365 | 53.1% | 0.0% |
| Projectile-only selector | 0.365 | 53.1% | 0.04% |
| Enemy-only selector | 0.215 | 87.7% | 34.7% |
| Unified enemy + projectile selector | 0.215 | 87.7% | 34.7% |
| Unified selector + switch penalty | 0.216 | 84.1% | 34.5% |

The unified selector has the lowest modeled risk. The switch-penalty variant
loses very little risk reduction while reducing direction switches by about
13.5%, so it is the preferred runtime shape.

Across 250 ms and 500 ms damage horizons, the recording showed that enemy
movement/contact states preceded approximately 91.5% and 79.9% of damage
samples respectively. Projectile-only unavoidable states preceded less than
0.1% of damage samples. These are repeated time samples, not unique causal
hits. “Unavoidable” means all nine tested movement actions intersected the
predicted hazard envelope; it does not prove that the game made the hit
strictly impossible to avoid.

## Runtime decision architecture to retain

The active runtime should have exactly one action-resolution pipeline:

```text
policy proposal
      |
      v
unified hazard assessment
      |
      v
one hazard override, if the risk margin is material
      |
      v
explicit crowd-recovery mode, only under emergency conditions
      |
      v
one final action writer -> bridge
```

### Responsibilities

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| Learned policy | Propose the action that best serves movement, combat, loot, and survival objectives | Keep |
| Unified hazard assessment | Score enemy motion/contact, projectiles, telegraphs, boss danger, and boundaries for every action | Keep; single source of truth |
| Hazard arbiter | Compare the requested action with the safest candidate using a threshold, margin, and switch penalty | Keep |
| Crowd recovery | Temporarily prioritize escape when crowd density or edge pressure is high; reuse the unified score | Keep as an explicit emergency mode |
| Raw recorder | Capture high-rate observations independently of control and training | Keep separate |
| Reward engine | Score the observed transition; do not secretly rewrite the action | Keep separate |
| Menu/build automation | Handle non-combat UI decisions through its own interface | Keep separate |
| Standalone projectile selector | Duplicate runtime action arbitration for one hazard type | Remove from runtime path |
| Standalone enemy-contact veto | Duplicate enemy path arbitration before the unified score | Remove from runtime path |

The standalone selector classes may remain temporarily as compatibility-tested
utilities while the migration is reviewed, but the environment must not call
them as additional action writers.

### Required decision trace

Every control step should expose one structured trace containing:

- requested action;
- final applied action;
- decision source: `policy`, `hazard`, or `crowd_recovery`;
- total requested and applied risk;
- enemy risk;
- projectile risk;
- telegraph/indicator risk;
- boundary risk;
- whether an override occurred;
- whether recovery mode is active;
- measured state/control interval.

The trace should be emitted once per control decision and should be usable by
both JSONL replay and TensorBoard. Avoid long dynamic metric names; use short,
stable tags under `combat/hazard_*`.

## Target repository layout

The final structure should make ownership obvious. The exact names can change,
but the boundaries should remain:

```text
src/brotato_ai/
  domain/
    actions.py          # action enum and movement vectors
    state.py            # normalized immutable state contract
    decisions.py        # policy proposal, hazard assessment, decision trace
  bridge/
    client.py           # protocol, handshake, state/action transport
    rate.py             # measured and requested rates
  control/
    policy.py           # learned policy interface
    hazards.py          # one unified hazard scorer
    arbiter.py          # one final action resolver
    recovery.py         # explicit crowd/edge emergency mode
  data/
    schema.py           # versioned JSONL record schema
    recorder.py         # independent high-rate recorder
    replay.py           # deterministic replay loader
    cache.py            # bounded cache with 10 GB enforcement
  training/
    ppo.py              # training entrypoint and model construction
    callbacks.py        # checkpoints and telemetry
    configs.py          # validated configuration
  evaluation/
    backtest.py         # same-trace controller comparisons
    metrics.py          # risk, damage, survival, switching metrics
    reports.py          # machine-readable and human-readable reports
  ui/
    build_policy.py
    menu_controller.py
tests/
  unit/
  replay/
  integration/
tools/
  launch/
  inspect/
```

Until the move is complete, the current v4 entrypoints and shared runtime
files remain the source of truth. Do not duplicate them into a second package
without adding an import-ownership map and running the full suite.

## Data contracts

Use explicit typed contracts instead of passing loosely shaped dictionaries
between every subsystem.

### `StateSnapshot`

Must contain normalized values for:

- timestamp, tick, session, and phase;
- arena dimensions;
- player position, velocity, radius, health, and maximum health;
- wave number and remaining time;
- enemies with stable runtime IDs, position, velocity, radius, boss flag, and
  semantic attack method;
- projectiles and telegraphs with stable owner IDs when available;
- bridge path-risk vectors for projectile, enemy, and boundary movement;
- counters and UI state needed by the active policy.

Missing optional fields must have documented defaults. A recorder record must
identify its schema version and must never silently change field meaning.

### `DecisionTrace`

One trace corresponds to one requested policy action and one final action sent
to the bridge. It must be possible to replay the same trace without starting
the game.

### Recorder/trainer separation

The recorder may sample close to 60 Hz and write asynchronously. The controller
may operate at a lower rate, but its actual rate must be measured. Training
must consume immutable records or a replay stream; it must not compete with
the bridge for ownership of the game action channel.

## Migration plan

### Phase 0 — Freeze the baseline

1. Stop the trainer normally so its checkpoint is saved.
2. Save the exact branch, commit, uncommitted diff, launcher settings, and
   active model path.
3. Preserve the raw recording and TensorBoard run used for the baseline.
4. Record baseline control Hz, recorder Hz, wave survival, damage, deaths,
   direction switches, hazard overrides, and victory rate.

### Phase 1 — Establish ownership

1. Add a short architecture/ownership document next to the code.
2. Identify every module that can send a movement action.
3. Enforce the rule that only the final arbiter may call the bridge action
   sender.
4. Remove unused environment settings instead of keeping dead feature flags.
5. Add one configuration object with validation and a printed startup summary.

### Phase 2 — Extract normalized state and traces

1. Normalize bridge state once at the boundary.
2. Convert normalized state to observation features in a separate adapter.
3. Define `DecisionTrace` and record it for live and replay execution.
4. Add schema-version checks and fixtures for missing/partial bridge payloads.

### Phase 3 — Build replay before moving behavior

1. Replay a fixed recording through the current policy proposal.
2. Replay the same recording through the unified hazard arbiter.
3. Compare policy-only, projectile-only, enemy-only, unified, and stable
   unified variants.
4. Produce a JSON report and a small summary table.
5. Do not accept a change based only on a single live run.

### Phase 4 — Move the code by boundary

Move one responsibility at a time: protocol, state normalization, hazard
assessment, arbitration, recorder, training, and evaluation. After each move:

- imports must have one owner;
- focused tests must pass;
- the replay report must remain within the agreed regression tolerance;
- the live bridge must not be needed for unit tests.

### Phase 5 — Reconnect live training

1. Start the game with mods enabled.
2. Confirm the bridge handshake and configured state rate.
3. Start the combined launcher only after confirming no trainer owns port 4242.
4. Confirm the first telemetry window contains `combat/hazard_*` metrics.
5. Monitor the first bounded run before allowing unattended training.

## Evaluation gates for every future feature

No new controller feature should be merged into live training until it has:

1. a stated owner and a single call site;
2. a typed input/output contract;
3. a unit test for normal, missing-data, and conflict cases;
4. a same-recording replay comparison against the current baseline;
5. override rate, risk reduction, switching, and damage metrics;
6. a feature flag with a safe default;
7. compact telemetry with stable names;
8. a clear rollback path;
9. a live smoke test after a normal restart.

The minimum comparison set for movement changes is:

- policy only;
- current unified hazard structure;
- proposed structure;
- proposed structure with switch penalty or hysteresis;
- a no-op control to detect analyzer drift.

Report both aggregate samples and unique damage windows. Repeated health-loss
samples must not be presented as unique hits or proof of causality.

## Operational rules

- Use `train.bat` for the combined recorder/training workflow when that is the
  intended launcher.
- Use `close_v4_temporal_scheduled.bat` for a normal stop and checkpoint save.
- Use `record_v4_raw.bat` for high-rate recording.
- Use `build_v4_raw_cache.bat` only as a bounded, background data task; it must
  never delay trainer startup.
- Never start a second trainer while port 4242 is occupied.
- Keep raw recording and action control as separate processes and ports.
- Keep the asset library below 10 GB; enforce the limit in code with an
  observable cleanup policy rather than relying on manual deletion.
- Do not use SB3 `time/fps` as the game control-rate metric. Use the measured
  `control/effective_state_hz` stream.
- Treat the Python controller path as CPU-sensitive. CUDA is for model
  training/inference batches where it helps; moving every small hazard
  calculation through CPU/GPU transfers is likely to hurt latency.
- Do not edit or overwrite `C:\ml\brotato\v1\shop\ocr_winmedia.py`.

## Definition of done

The restructure is complete when:

- one active package owns the runtime;
- one arbiter owns all action writes;
- the hazard breakdown is inspectable in replay, JSONL, and TensorBoard;
- recorder and controller run independently;
- control-rate and recorder-rate measurements are separate;
- the cache enforces the 10 GB limit;
- a fresh checkout can run the tests without the game;
- replay reports are deterministic and compare controller variants;
- the full test suite passes;
- a normal restart activates the new source and the handshake is verified;
- a bounded live run shows no duplicate trainer, no stale TensorBoard source,
  and no unexplained action overrides.

## First actions for the restructuring owner

1. Stop the current trainer normally and preserve its checkpoint.
2. Create a dedicated restructure branch from the exact current worktree.
3. Add the ownership map and state/decision contracts before moving files.
4. Move the replay analyzer into the repository’s evaluation area and make it
   a maintained test tool.
5. Refactor one boundary at a time, running the full suite and same-trace
   backtest after each boundary.

