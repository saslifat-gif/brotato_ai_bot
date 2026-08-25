# Brotato RL Bridge Protocol v1

The game mod is a TCP client. The Python trainer listens only on
`127.0.0.1:4242`. Each message is one UTF-8 JSON object followed by `\n`.
Every message contains `"protocol": 1` and a string `type`.

Game to trainer:

- `hello`: mod/game versions and capabilities.
- `state`: tick, phase (`combat`, `wave_end`, `item_found`, `shop`, `upgrade`,
  `game_over`, `victory` or `menu`), player, arena, wave, counters, enemies, projectiles,
  pickups, build metadata, available structured UI actions, death and victory fields. Shop,
  upgrade and found-item actions include a language-independent `choice` object with internal
  IDs, category, tier, price, affordability, tags and numeric effects. Full inventory/build
  metadata is emitted only during UI phases. Combat states contain a compact `combat` summary
  with character, weapon counts/range, movement speed, armor and attack speed. Bridge 0.3.3
  adds per-weapon IDs, range, cooldown/reload timers, ammo capacity, readiness and attack state.
  Enemy entries advertise stable IDs/types, exact collision width/height when available,
  contact damage, attack/movement type, cooldown, charging/attacking and boss/elite flags.
  Pickup entries advertise stable identity, collision size, and a `category` of `healing`,
  `crate`, `material`, or `consumable`, plus healing/material values where the game exposes
  them. `attack_indicators` contains visible API-discovered warning geometry, direction,
  activation time and damage. When red telegraphs are represented as hostile
  projectile nodes, they are mirrored into this channel with
  `source: "projectile"`. Projectile entries include ID, collision geometry,
  damage, lifetime and attack type. Bridge 0.2.1+ also emits
  `human_action`, the nine-way action returned by Brotato's vanilla movement behavior, plus
  `human_input_age_ms`. Observation-only recorders can capture keyboard demonstrations without
  sending combat actions. `sequence` acknowledges the most recent
  action applied by the bridge, so queued old states are never used as a new
  training step.
- `event`: non-state notification such as `manual_reset_required`.
- `error`: rejected command or protocol error.

Trainer to game:

- `action`: sequence and discrete movement action `0..8`.
- `ui_action`: activates one currently visible and enabled game button by the
  exact node identifier advertised in the latest state.
- `reset`: releases movement control before the trainer uses an advertised
  restart action. It remains safe when no restart action is visible.
- `configure`: requests a structured state rate from 4 to 24 Hz. Human
  demonstration recording requests 8 Hz; PPO keeps the adaptive maximum.

Movement actions are idle, up, down, left, right, up-left, up-right,
down-left and down-right. Combat runs in real time at up to 24 structured
observations per second, adapting to 16 from wave 6 and 12 from wave 10 to
protect late-wave game frame time. An action remains active between state
updates and expires after 1.5 seconds without a trainer update. Normal human
input is restored immediately after expiry or a disconnect. The trainer
resends its last action after a reconnect and accepts a new low tick value when
Brotato itself has restarted.
