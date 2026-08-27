# V4 Runtime Ownership

## Action-resolution contract

```text
learned policy proposal
        -> one unified hazard assessment
        -> one material-risk override
        -> explicit crowd/edge recovery, reusing the same score
        -> FinalActionWriter
        -> bridge port 4242
```

`FinalActionWriter.write()` is the only production movement call site. `BridgeClient.send()` rejects `type=action`, so a legacy selector cannot become a second action writer by accident.

## Import owners

| Responsibility | Active owner | Compatibility boundary |
| --- | --- | --- |
| Actions and vectors | `brotato_ai.domain.actions` | `v3.protocol.MoveAction` |
| Normalized state | `brotato_ai.domain.state` | None; normalize once in the environment |
| Decision trace | `brotato_ai.domain.decisions` | Existing info fields remain stable |
| Bridge transport | `brotato_ai.bridge.client` | `v3.bridge_server` |
| Measured rates | `brotato_ai.bridge.rate` and environment telemetry | `control/effective_state_hz` |
| Unified hazards | `brotato_ai.control.hazards` | Legacy v3 selector classes are test utilities only |
| Final arbitration/write | `brotato_ai.control.arbiter` | `v3.env.brotato_api_env` calls it once |
| Raw schema/recorder | `brotato_ai.data.schema`, `brotato_ai.data.recorder` | `v3.record_raw` |
| Bounded storage | `brotato_ai.data.cache` | None |
| Replay | `brotato_ai.data.replay` | None |
| PPO launcher | `brotato_ai.training.ppo` | Temporal model remains import-compatible under `v4` |
| Evaluation/reporting | `brotato_ai.evaluation` | None |
| Menu/build UI | `brotato_ai.ui` | Existing v3 implementations remain isolated from movement arbitration |

## State boundary

Every bridge state is normalized once into an immutable `StateSnapshot`. Missing optional values use documented defaults:

- arena: 1920 by 1080;
- player position: arena center when absent; velocity: zero;
- player radius: 28;
- player maximum health: at least 1;
- phase: `unknown`;
- tick/timestamp: -1;
- collections: empty;
- all nine path-risk values: zero.

Unknown source fields remain available in the immutable payload, so observation adapters can consume existing counters, combat fields, and UI state without changing core field meanings.

## Recorder and controller separation

- Port 4242 is owned by the trainer/control process.
- Port 4243 is owned by the independent recorder.
- Recorder writes are queued to a background writer thread.
- Trainer startup only reads the last completed raw-anchor cache; refresh runs independently.
- Control Hz, recorder Hz, and SB3 throughput are distinct telemetry streams.

## Legacy utilities

`ProjectileHazardSelector` and `EnemyContactGuard` may remain during review for compatibility tests and replay variants. The active environment does not import or call them. They do not own the bridge.
