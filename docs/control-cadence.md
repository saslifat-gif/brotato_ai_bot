# Control cadence correction (bridge 0.3.24)

Every movement message previously reset `_state_elapsed` to zero. Thus the
next acknowledged observation waited another publication interval after the
controller's inference and processing time. At a requested 24 Hz, a 300-step
live diagnostic averaged 62.97 ms between decisions and 48.46 ms in state wait.

Movement messages now preserve the publication clock. Rich-state publication
waits until `Engine.get_physics_frames()` advances beyond the frame in which
the action was accepted, ensuring a physics opportunity before acknowledgement.
Overdue clock time uses modulo to avoid bursts of catch-up publications.
Raw recorder states include the current and last-action physics frame counters.
The raw recorder remains independent and does not promise post-action physics.

Godot API: https://docs.godotengine.org/en/3.5/classes/class_engine.html#class-engine-method-get-physics-frames

Live validation on Windows, 300 decisions per run, same model, 24 Hz requested,
one Torch thread, shadow human policy:

| Metric | Before | After |
| --- | ---: | ---: |
| Mean decision interval | 62.97 ms | 43.70 ms |
| Mean source interval during combat | 63.19 ms | 43.87 ms |
| Mean state wait | 48.46 ms | 27.66 ms |
| Mean decision pipeline | 5.66 ms | 6.37 ms |

The after profile includes time spent waiting in game menus, so its whole-process
throughput is not a combat throughput measure. The table uses per-step timing.
The runs are not deterministic replays; these are timing checks, not a controlled
survival or win-rate comparison. All 215 Python regression tests passed. The
live bridge handshake confirmed 0.3.24 and completed all 300 steps.
