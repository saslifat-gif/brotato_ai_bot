# Source-boundary diagnosis

The active bridge is a persistent, push-based TCP stream. It does not expose
a request/poll operation that a faster Python, Rust, or C++ reader could call
more frequently. The mod publishes a raw stream on port `4243` and the
controller stream on port `4242`.

## Existing recording result

`models/version_3/raw_records/rate_experiment.jsonl` contains 70,472 received
rows. The preserved game-side `published_at_ms` and recorder-side
`recorded_at_ms` produce:

| Measurement | Mean interval | Median | p95 | Effective rate |
| --- | ---: | ---: | ---: | ---: |
| game publication timestamp | 16.784 ms | 17 ms | 19 ms | 59.581 Hz |
| recorder arrival proxy | 16.784 ms | 17 ms | 19 ms | 59.580 Hz |

There are 70,469 fresh states and only 3 repeated source rows. By projectile
count, fresh-state rates were 59.508 Hz for 0–10 projectiles, 59.889 Hz for
11–25, and 59.820 Hz for 26–50. No projectile bucket showed a drop toward
10 Hz.

The earlier `10.07 Hz` dense-projectile figure was an analysis artifact: the
benchmark selected noncontiguous high-projectile rows and incorrectly treated
the gaps between selected rows as source intervals. `v3/profile_runtime.py`
now labels those intervals as noncontiguous subset gaps and no longer reports
them as a source rate.

## Repeatable diagnostics

Analyze an existing recording:

```text
PYTHONPATH=src;. python v3/diagnose_source.py models\version_3\raw_records\rate_experiment.jsonl --output reports\source_boundary_existing_recording.json
```

Run the minimal reader against the independent raw stream. It performs no
controller, feature, reward, action, or disk-recording work:

```text
PYTHONPATH=src;. python v3/diagnose_source.py --seconds 60 --output reports\source_boundary_minimal_live.json
```

The normal trainer can additionally write bounded per-state 4242 boundary
samples by setting `BROTATO_RUNTIME_PROFILE_PATH`. Those samples include local
socket arrival, payload completion, JSON parsing, processing start/end, action
decision, source timestamp/sequence, payload size, and entity counts.

## Current conclusion

The available evidence rejects the hypothesis that the observed ~10 Hz is a
game/API fresh-state limit. It also rejects a raw-recorder ingestion limit:
the local arrival cadence matches the game publication cadence. The current
24 Hz controller stream is an intentional control-rate schedule, not evidence
that the source can only produce 10 Hz.

A live 4242 profile is still useful for attributing any delay inside the full
controller loop, but a native ingestion rewrite is not justified by the
existing source-boundary evidence.

## Where 24 Hz comes from

The controller cadence is a deliberate publisher schedule, not a hidden
socket or Python polling limit:

1. `v3/mod/Lifat-BrotatoRLBridge/bridge.gd::_process` accumulates
   `_state_elapsed` and calls `_publish_state` when it reaches
   `_state_interval_sec()`.
2. `_state_interval_sec()` returns `1.0 / clamp(_requested_state_hz, 4, 60)`.
3. `_requested_state_hz` starts at `DEFAULT_STATE_HZ = 24.0`, is reset to 24
   on connection, and is set by the trainer's `configure` message.
4. The Python protocol/configuration and the trainer CLI also default to 24;
   before the sensitivity experiment they duplicated a 24 Hz upper bound.

The code and documentation describe 24 Hz as a conservative rich-state budget
to leave the game's main thread enough time for state construction and to avoid
feeding queued observations to the controller. The earlier implementation
also reduced late-wave control to 16/12 Hz; that adaptive cap was removed
because it slowed control precisely during boss projectile pressure. There is
no benchmark or test in the repository that demonstrates that 24 Hz is a
necessary maximum. The current source measurements therefore justify testing
the schedule, not assuming it is the bottleneck.

The client now drains complete buffered state messages and returns the newest
state that satisfies `after_tick`, `minimum_sequence`, and `combat_only`.
Actions are still held by the mod between updates and expire after 1.5 seconds
without a new action. This preserves the real action-persistence behavior
while preventing TCP backlog from adding artificial queue latency.
