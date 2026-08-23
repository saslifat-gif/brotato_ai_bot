# Brotato AI v2

V2 is a detector-driven agent. It does not feed screenshots directly into PPO
and it does not use hard-coded menu coordinates.

```text
MSS capture -> YOLO26n + tracking -> 98-value combat state -> RecurrentPPO
                    |
                    +-> detected UI button boxes -> safe UI controller
```

V1 remains available and unchanged. V2 intentionally refuses to start agent
training until custom combat detector weights exist.

## 1. Record representative gameplay

Open Brotato and manually play several runs covering early/late waves, shops,
upgrades, item pickups and death screens. Then run:

```bat
record_v2.bat
```

The recorder restores the Brotato window, waits three seconds, then saves 10
frames per second plus the human WASD action under `datasets/v2/raw/session_*`.
Start it while a battle is ready to play and play normally. To stop safely,
Alt+Tab to the recorder console and press `Q` or `Enter`. There is no global
stop hotkey, so the stop command cannot also trigger an action inside Brotato.
V2 uses Windows Graphics Capture by default because it produces fresh frames
for the game on the primary monitor. At the end, `visual_change_ratio` should
be well above zero. If the recorder reports that capture appears frozen, do
not label that session.

Curate the recording before labeling:

```bat
curate_v2.bat
```

The visual curator samples every tenth frame. Press `C` for a combat frame,
`U` for a shop/upgrade/item/death UI frame, or `S` to skip it. Use `N` and `P`
to inspect adjacent raw frames. Selected images are copied into
`datasets/v2/to_label/combat` and `datasets/v2/to_label/ui`.

Near-duplicate skipping is optional. To enable it for a mostly static
recording, run `python -m v2.curate_recording --min-change 2`.

## 2. Label two datasets

Use CVAT, Roboflow, or another YOLO-compatible annotation tool. SAM 2 can help
propagate object labels through adjacent recorded frames.

Combat classes:

```text
player, enemy, projectile, loot, obstacle
```

UI classes:

```text
restart_button, take_item_button, upgrade_card,
next_wave_button, shop_card, refresh_button
```

Export YOLO detection datasets into:

```text
datasets/v2/combat/images/train
datasets/v2/combat/images/val
datasets/v2/combat/labels/train
datasets/v2/combat/labels/val

datasets/v2/ui/images/train
datasets/v2/ui/images/val
datasets/v2/ui/labels/train
datasets/v2/ui/labels/val
```

Do not split neighboring frames randomly between train and validation. Keep
whole recorded sessions in one split so validation measures new gameplay.

## 3. Train the detectors

For an NVIDIA GPU, use `--device 0`. Use `--device cpu` otherwise.

```powershell
python -m v2.train_detector --task combat --epochs 100 --device 0
python -m v2.train_detector --task ui --epochs 100 --device 0
```

The best weights are copied automatically to:

```text
models/version_2/combat_best.pt
models/version_2/ui_best.pt
```

## 4. Validate before RL

The combat detector must consistently find the player and nearby threats. The
UI detector must have extremely few false positives, especially for restart,
upgrade and next-wave buttons. A missed UI button is safer than a wrong click.

```bat
validate_v2.bat
```

To validate UI weights instead:

```powershell
python -m v2.validate_detector --task ui
```

## 5. Train the recurrent agent

```bat
train_v2.bat
```

The observation contains normalized player position/visibility, HP, nearest
enemy vectors, projectile density and distance by direction, nearest loot,
obstacle distance, and the previous action. The policy uses an LSTM so it can
reason across consecutive frames.

## Safety boundaries

- UI actions require a detected button box with confidence at least 0.65.
- Restart has priority over every other menu action.
- Shop purchases are disabled until currency/price perception is implemented.
- Missing detector weights cause a clear startup error instead of falling back
  to legacy templates or random clicks.
