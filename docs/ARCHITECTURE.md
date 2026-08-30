# Brotato AI V4 architecture

There is one active runtime. `src/brotato_ai` owns shared contracts and
control primitives; `v4` owns the game-facing runner, bridge server, training
and evaluation entrypoints, human-demo tools, and the Brotato mod.

## Runtime

Every action follows one path:

```text
policy proposal
  -> unified hazard assessment
  -> material-risk override
  -> emergency recovery, when required
  -> one final action writer
  -> Brotato bridge
```

Only `brotato_ai.control.arbiter`/`FinalActionWriter` may write a movement
action. The learned human policy is proposal-only in `SHADOW_HUMAN`; it is not
production control. `HYBRID_HUMAN` stays disabled until unseen shadow evidence
passes the safety and generalization gates.

## Ownership

| Responsibility | Owner |
| --- | --- |
| actions, state, decision traces | `src/brotato_ai/domain` |
| bridge transport and rates | `src/brotato_ai/bridge`, `v4/bridge_server.py` |
| hazard scoring and arbitration | `src/brotato_ai/control` |
| learned human-policy adapter | `src/brotato_ai/policy` |
| SQLite/JSONL recording and replay | `src/brotato_ai/data`, `v4/record_*.py` |
| validated runtime configuration | `src/brotato_ai/training/configs.py`, `v4/config.py` |
| build/menu decisions | `src/brotato_ai/ui`, `v4/ui_*.py` |
| training and frozen execution | `v4/train*.py`, `v4/run_frozen.py` |
| DAgger selection and review | `v4/dagger_*.py` |
| game mod | `v4/mod/Lifat-BrotatoRLBridge` |

## Modes

- `HANDCRAFTED` (default): existing production behavior; no learned human
  inference runs.
- `SHADOW_HUMAN`: handcrafted actions remain applied while learned proposals,
  confidence, risk, safety, and agreement are logged.
- `HYBRID_HUMAN`: experimental and currently disabled.
- `EXPERIMENTAL_FULL_LEARNED`: explicit offline/diagnostic opt-in only.

## Data boundaries

The human recorder is observation-only. It records raw input, rich state,
derived features, rewards, action segments, build choices, fixed-horizon
outcomes, and exact episode timestamps. F9 is an observation bookmark; it does
not change input or create a synthetic label.

DAgger corrective labels must come from a real human decision at a selected
bot-state or from a manually played state. The system must never infer a human
label from the model proposal, safety arbiter, or counterfactual risk score.

## Compatibility

The old detector-era v1/v2 implementations and their launchers are removed
from the active project. The previous API runtime has been absorbed into v4;
there is no second runtime package. Existing checkpoint directories under
`models/version_3` are retained as data compatibility paths only.

The user-owned `v1/shop/ocr_winmedia.py` file is intentionally left untouched
and is not imported by the V4 runtime. Runtime configuration and launchers use
only `BROTATO_V4_*` names.
