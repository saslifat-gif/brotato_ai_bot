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
trainer disconnects. During training, combat runs step-by-step: Brotato pauses
after each observation and resumes when the policy returns its next action.
This prevents PPO updates from leaving the character idle while the real-time
game continues.

Automatic shop, upgrade and run-reset API hooks are intentionally not guessed.
The initial adapter reports menu phases and waits while you handle them
manually. Once `diagnose_v3.bat` reports your installed game version and scene
state correctly, those version-specific hooks can be added safely.

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

Open Brotato, enable **BrotatoRLBridge** in the Mods menu, close the game and
launch it again. Do not publish this training bridge to Workshop.

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

Run `train_v3.bat`, launch Brotato and enter a wave. During this first adapter
stage, manually handle shops, upgrades, next-wave screens and restarting after
death. Training pauses safely until combat resumes.

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
