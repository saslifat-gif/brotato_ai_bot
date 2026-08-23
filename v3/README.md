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
<Brotato>\mods\Lifat-BrotatoRLBridge-0.1.1.zip
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

## API observation and actions

The observation is a fixed 256-value vector containing player status, wave and
arena values, the nearest 24 enemies, 16 projectiles and 14 pickups. The policy
has nine movement actions: idle, four cardinal directions and four diagonals.

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

The Stick/Melee profile ranks internal item IDs and numeric effects rather than
localized screen text. It prioritizes Stick weapons plus melee/attack upgrades,
masks unaffordable purchases, and records structured decisions to
`models/version_3/ui_decisions.jsonl`. Train the compact reusable scorer after
collecting decisions:

```powershell
python -m v3.train_ui_build --dataset models/version_3/ui_decisions.jsonl --output models/version_3/ui_build_base.pt
```
