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
real-time at up to 24 structured observations per second; global scene pauses
are deliberately avoided because they can interrupt Brotato's wave-cleanup
signals.

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
<Brotato>\mods\Lifat-BrotatoRLBridge-0.2.1.zip
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
`models/version_3/human_combat_v1.jsonl` with a bridge session and episode ID.
This is structured state/action data, not screen video. Train/validation splits
keep complete episodes together to prevent adjacent-frame leakage.

After collecting several complete runs and at least 10,000 records, train the
compact behavior-cloning base:

```powershell
python -m v3.train_combat_bc --dataset models/version_3/human_combat_v1.jsonl --output models/version_3/human_combat_base.pt
```

## API observation and actions

The observation is a fixed 256-value vector containing player status, wave and
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
`models/version_3/ui_decisions_stick_melee_v2.jsonl`. Train the compact reusable
scorer only after collecting at least 200 valid item choices:

```powershell
python -m v3.train_ui_build --dataset models/version_3/ui_decisions_stick_melee_v2.jsonl --output models/version_3/ui_build_base.pt
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
