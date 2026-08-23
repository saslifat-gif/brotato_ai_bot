# Brotato RL Bridge Protocol v1

The game mod is a TCP client. The Python trainer listens only on
`127.0.0.1:4242`. Each message is one UTF-8 JSON object followed by `\n`.
Every message contains `"protocol": 1` and a string `type`.

Game to trainer:

- `hello`: mod/game versions and capabilities.
- `state`: tick, phase (`combat`, `wave_end`, `shop`, `upgrade`, `game_over`,
  `victory` or `menu`), player, arena, wave, counters, enemies, projectiles,
  pickups, available structured UI actions, death and victory fields. `sequence` acknowledges the most recent
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

Movement actions are idle, up, down, left, right, up-left, up-right,
down-left and down-right. Combat runs in real time at up to 15 structured
observations per second. An action expires after 1.5 seconds without a trainer
update, and normal human input is restored immediately after expiry or a
disconnect. The trainer resends its last action after a reconnect and accepts
a new low tick value when Brotato itself has restarted.
