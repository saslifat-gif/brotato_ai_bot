# Brotato AI v3 — Local Training API

V3 replaces screen capture, OCR and physical clicks with a local Brotato mod.
The mod exports structured game state and accepts movement actions through a
versioned JSON-lines connection on `127.0.0.1:4242`. It never listens on the
LAN or internet.

V1 and v2 remain intact. Back up `%APPDATA%\Brotato` before enabling any mod.
Brotato's current PC release already includes Godot Mod Loader; do not install
the archived standalone Brotato-ModLoader package.

## Current scope

The first adapter trains combat movement from exact player, enemy, projectile,
pickup, wave and arena state. Normal keyboard control is restored whenever the
trainer disconnects or stops sending actions for 1.5 seconds. Combat remains
real-time at up to 24 structured observations per second. The bridge adapts to
16 observations per second from wave 6 and 12 from wave 10, leaving the game
more frame time when enemy and projectile counts become dense. Actions remain
active between observations. Global scene pauses are deliberately avoided
because they can interrupt Brotato's wave-cleanup signals.

The adapter advertises visible, enabled game UI actions through the structured
API. Training automatically chooses an upgrade, makes a bounded number of shop
purchases/rerolls, starts the next wave, and activates the verified retry-wave
control after death. It emits the game buttons' own signals, so Brotato still
enforces affordability, disabled states, purchase logic, and saves.

## 1. Install the local bridge mod

Run:

```bat
install_v3_mod.bat
```

Select the folder containing `Brotato.exe`. The installer creates the runtime
package where the exported game discovers local mods:

```text
<Brotato>\mods\Lifat-BrotatoRLBridge-0.3.8.zip
```

It also keeps an editable diagnostic copy under
`<Brotato>\mods-unpacked\Lifat-BrotatoRLBridge`. The ZIP contains the required
`mods-unpacked\Lifat-BrotatoRLBridge\...` internal structure.

For Steam installations, it additionally copies the ZIP beside an existing
subscribed Workshop mod:

```text
<Steam>\steamapps\workshop\content\1942280\<numeric-workshop-id>
```

This is required because Brotato's bundled ModLoader only scans subscribed
numeric Workshop directories. The installer also activates **BrotatoRLBridge**
in the current ModLoader profile when that profile already exists.

In Steam, open **Brotato → Properties → General → Launch Options** and add
`--enable-mods`. Launch the game, then restart Brotato once more.

If the installer says the profile was not found, enable **BrotatoRLBridge** in
the Mods menu after the first mod-enabled launch and restart. Do not publish
this training bridge to Workshop.

## 2. Verify the API before training

Run `diagnose_v3.bat`, then start/restart Brotato and enter a wave. A healthy
bridge prints changing ticks plus HP, enemy, projectile and pickup counts. It
does not open a screenshot or move the mouse.

If Brotato reports a mod error, send these files/output before training:

```text
%APPDATA%\Brotato\logs\modloader.log
diagnose_v3.bat output
```

## 3. Train movement

Run `train_v3.bat`, launch Brotato and enter the first wave. After that, the
verified shop, upgrade, next-wave and retry-wave screens are automated.

Models and TensorBoard logs are written under `models/version_3`.
Press `Ctrl+C` in the training console to save the interrupted model and
disconnect the bridge; Brotato then unpauses and restores normal keyboard input.

## Record human combat demonstrations

Run `record_v3_human.bat` when the bridge is installed. Python sends no combat
actions in this mode, so Brotato's normal WASD movement remains authoritative.
The structured Stick/Melee teacher handles shops, upgrades, found items, next
wave and restart controls while you play the waves.

Human movement is sampled at 8 Hz, immediate direction changes are retained,
and repeated idle input is limited to 2 Hz. Records are written to
`models/version_3/human_semantic_combat_v2.jsonl` with a bridge session and episode ID.
This is structured state/action data, not screen video. Train/validation splits
keep complete episodes together to prevent adjacent-frame leakage.

Bridge 0.3.3 records an 832-value API-only semantic observation. It preserves
the prior 384 rich inputs and adds stable enemy identity and collision size,
pickup category/healing/value, per-weapon cooldown/reload/ammo readiness, and
visible attack-warning geometry. No screen capture or CV is used.

After collecting several complete runs and at least 1,000 new semantic records,
warm-start the compact semantic base from the existing human combat base:

```powershell
python -m v3.train_semantic_combat_bc --dataset models/version_3/human_semantic_combat_v2.jsonl --base-model models/version_3/human_combat_base_candidate.pt --output models/version_3/semantic_combat_base_candidate.pt
```

The semantic residual is initialized to zero, so before semantic training its
actions exactly match the old human base. The saved model remains small (under
100,000 parameters), and its loss/validation curves are written to
`models/version_3/logs/SemanticCombatBC` for TensorBoard.

After validating that base, run `train_v3_semantic_rl.bat` for live PPO
fine-tuning. PPO starts with exactly the semantic base's action logits, uses the
832-value API observation at 12 Hz, and applies a small supervised anchor after
each rollout to limit catastrophic forgetting. It saves periodic checkpoints
under `semantic_finetune_checkpoints`, the best rolling-reward model under
`semantic_finetune_best`, and TensorBoard curves under `SemanticBasePPO`.
Safety overrides are disabled by default because silently replacing a sampled
action would invalidate PPO's on-policy update.

## Whole-arena policy generation

`train_v3_full_arena_rl.bat` migrates a trained semantic PPO checkpoint to a
1,512-value observation. Its first 832 values are unchanged. The appended
inputs contain a 10 by 6 whole-arena map for enemy density/danger/motion,
projectile density/damage/motion, healing pickups and material/crate pickups,
plus exact charge direction and attack-target geometry for the nearest 20
enemies. Bridge 0.3.8 computes the enemy map over every live enemy before the
detailed API list is capped, so dense late waves remain visible without the
cost of serializing every enemy object.

The migration copies the complete trained semantic PPO actor and its action
head. The new residual is initialized to zero and the trainer verifies action
logits before saving the bootstrap checkpoint. Consequently the generation
starts with exactly the source model's movement behavior while gaining new
inputs it can learn to use. Existing human records are padded safely and remain
the behavior-cloning anchor; a new recording is optional, not required.

By default the batch file reads
`semantic_finetune_best/best_training_agent.zip`. To migrate a newer checkpoint,
run:

```powershell
python -m v3.train_full_arena_finetune --source-model models/version_3/semantic_finetune_checkpoints/semantic_base_ppo_200000_steps.zip --state-hz 12
```

Add `--bootstrap-only` to verify and save the migrated checkpoint without
opening port 4242 or interrupting an existing semantic training process.

Checkpoints are written under `full_arena_finetune_checkpoints`, the best model
under `full_arena_finetune_best`, and TensorBoard curves under `FullArenaPPO`.

## Bullet-hell future-path generation

`train_v3_bullet_hell_rl.bat` migrates a trained full-arena PPO actor to a
3,941-value observation without changing its initial action logits. Bridge
0.3.8 computes a player-centered 20 by 12 danger map from every live hostile
projectile before the detailed projectile list is capped. The map includes
occupancy at now, 0.25, 0.5, 0.75 and 1.0 seconds; swept projectile radius;
separate horizontal/vertical direction lanes; and damage weighting.

The bridge also predicts collision risk for all nine movement actions,
separately for projectiles, enemy contact, and arena boundaries. The
bullet-hell environment adds
a small reward penalty when the sampled action chooses a dangerous path. It
does not silently replace PPO actions, so training remains on-policy while the
actor learns to leave stationary deadlocks and maintain safer spacing.

Migrate a current full-arena checkpoint offline first:

```powershell
python -m v3.train_bullet_hell_finetune --source-model models/version_3/full_arena_ppo_recovery.zip --bootstrap-only
```

Then run `train_v3_bullet_hell_rl.bat`, or resume
`bullet_hell_ppo_bootstrap.zip`. TensorBoard writes the new generation under
`BulletHellPPO`.

During each PPO gradient update, bridge 0.3.8 pauses the Godot scene and stops
publishing states. It resumes immediately after the update. This prevents the
1.5-second action timeout from handing control back to an idle human input while
enemies continue moving, and keeps unrecorded movement out of the rollout.

The bullet-hell environment also measures real displacement between structured
states. It removes the normal survival reward from IDLE steps, adds a small
low-motion penalty when a movement command produces almost no displacement, and
penalizes rapid opposite-direction reversals only when immediate projectile and
enemy-contact risk is low. TensorBoard exposes these under `movement/` so a
stationary or oscillating policy is visible instead of being hidden by aggregate
action counts.

## V4 temporal hierarchical movement

`train_v4_temporal_rl.bat` upgrades a trained 3,941-input bullet-hell actor to
a 4,077-input temporal actor. The complete V3 action function is copied with
exact initial logit parity, so migration does not erase the movement already
learned. A GRU reads the last eight action/displacement/damage/threat
transitions, while a transparent macro planner advertises one of five goals:
evade, heal, loot, engage, or reposition.

Verify the migration without touching a live V3 trainer:

```powershell
python -m v4.train_temporal_hierarchical --bootstrap-only
```

The human behavior anchor keeps every moving demonstration but limits IDLE to
about 10 percent, preventing stationary demonstrations from dominating V4.
Live training defaults to 20 Hz, writes checkpoints under
`v4_temporal_checkpoints`, and adds `v4/*` objective, urgency, and history
curves to TensorBoard under `V4TemporalPPO`.
`train_v4_temporal_scheduled.bat` is the unattended launcher: after a restart
it automatically selects the newest V4 checkpoint, falling back to the
verified bootstrap when no live checkpoint exists yet.

Fine-tune that human base online without discarding its behavior:

```bat
train_v3_human_base.bat
```

This uses a rich 384-value PPO environment initialized exactly from the compact
human actor. A small behavior-cloning update follows every PPO rollout to limit
catastrophic forgetting. TensorBoard writes the run as `HumanBasePPO` and adds
game-specific `combat/*`, `actions/*`, and `human_bc/*` curves alongside SB3's
standard reward and optimization metrics.

## API observation and actions

The legacy PPO observation is a fixed 256-value vector containing player status, wave and
arena values, the nearest 24 enemies, 16 projectiles and 14 pickups. The policy
has nine movement actions: idle, four cardinal directions and four diagonals.
This legacy vector remains unchanged so existing RecurrentPPO checkpoints stay
loadable. The next-generation behavior-cloning base uses a separate 384-value
rich vector with combat build information, enemy threat state, projectile
radius and time-to-impact.

Protocol details are in `v3/PROTOCOL.md`.

## Environment variables

- `BROTATO_V3_HOST` — defaults to `127.0.0.1`.
- `BROTATO_V3_PORT` — defaults to `4242`.
- `BROTATO_V3_TIMESTEPS` — defaults to `1000000`.
- `BROTATO_V3_DEVICE` — SB3 device, default `auto`.
- `BROTATO_V3_RESUME_MODEL` — optional checkpoint path.
- `BROTATO_V3_RESET_TIMEOUT` — how long training waits for manual combat resume.
- `BROTATO_V3_AUTOMATE_MENUS` — defaults to `1`; set `0` for manual menus.
- `BROTATO_V3_MAX_SHOP_BUYS` — maximum purchases per shop, default `4`.
- `BROTATO_V3_MAX_SHOP_REROLLS` — maximum rerolls per shop, default `1`.
- `BROTATO_V3_UI_BUILD_PROFILE` — structured UI teacher; defaults to `stick_melee`.
- `BROTATO_V3_UI_DATASET` — JSONL decision log used to train the small UI Build Base.
- `BROTATO_V3_UI_MODEL` — optional trained UI Build Base checkpoint; the rule teacher remains a fallback.
- `BROTATO_V3_SAFETY_SHIELD` — imminent-collision override; defaults off during PPO training.
- `BROTATO_V3_COMBAT_DATASET` — optional rich structured combat decision log.
- `BROTATO_V3_RESUME_LR` — resumed PPO learning rate; defaults to `0.00005`.
- `BROTATO_V3_RESUME_ENT_COEF` — resumed PPO entropy coefficient; defaults to `0.002`.

The Stick/Melee profile ranks internal item IDs and numeric effects rather than
localized screen text. It prioritizes Stick weapons plus melee/attack upgrades,
masks unaffordable purchases, and records versioned structured decisions to
`models/version_3/ui_decisions_stick_melee_v3.jsonl`. Train the compact reusable
scorer only after collecting at least 200 valid item choices:

```powershell
python -m v3.train_ui_build --dataset models/version_3/ui_decisions_stick_melee_v3.jsonl --output models/version_3/ui_build_base.pt
```

## Safe frozen collection and evaluation

Do not continue updating a strong combat checkpoint merely to collect UI data.
Run it deterministically with the geometric safety shield instead:

```powershell
.\collect_v3_safe.bat
```

This keeps the RecurrentPPO weights frozen, preserves LSTM state correctly,
automates menus, records UI decisions and writes rich combat examples for the
next combat base. For a bounded deterministic evaluation, run:

```powershell
python -m v3.run_frozen --model models/version_3/combat_peak_100883_agent.zip --episodes 10 --results models/version_3/eval_peak.json
```

Evaluate a behavior-cloned human combat base with the same UI automation and
per-episode safety metrics:

```powershell
python -m v3.run_frozen --policy bc --model models/version_3/human_combat_base_candidate.pt --episodes 10 --results models/version_3/eval_human_bc.json
```

The safety layer is intentionally disabled by default in `v3.train`: silently
replacing PPO actions would make its on-policy updates mathematically invalid.
Resumed PPO instead uses conservative defaults and saves the best rolling
training checkpoint under `models/version_3/best`. Deterministic frozen
evaluation remains the source of truth.

After collecting at least 10,000 rich combat decisions, train the small
behavior-cloning base:

```powershell
python -m v3.train_combat_bc --dataset models/version_3/combat_decisions_v1.jsonl --output models/version_3/combat_bc_base.pt
```
