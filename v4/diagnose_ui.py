"""Show structured Brotato UI actions whenever the phase or action set changes."""

import argparse

from v4.bridge_server import BridgeServer
from v4.config import load_config


def _signature(state):
    actions = state.get("ui", {}).get("actions", [])
    return (
        state.get("phase"),
        tuple(
            (a.get("id"), a.get("role"), a.get("enabled"), a.get("text"))
            for a in actions
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Brotato bridge UI actions")
    parser.add_argument("--states", type=int, default=1800)
    args = parser.parse_args()
    cfg = load_config()
    previous_tick = None
    previous_signature = None
    print("[v4-ui] play normally; actions are printed when a screen changes")
    with BridgeServer(cfg.host, cfg.port) as server:
        for _ in range(max(1, args.states)):
            state = server.wait_for_state(
                timeout_sec=cfg.reset_timeout_sec,
                after_tick=previous_tick,
            )
            previous_tick = int(state.get("tick", -1))
            signature = _signature(state)
            if signature == previous_signature:
                continue
            previous_signature = signature
            print(
                f"[v4-ui] tick={previous_tick} phase={state.get('phase')} "
                f"scene={state.get('scene')} materials="
                f"{state.get('counters', {}).get('materials')}"
            )
            last_result = state.get("ui", {}).get("last_result", {})
            if last_result:
                print(f"  last_result={last_result}")
            for action in state.get("ui", {}).get("actions", []):
                print(
                    "  "
                    f"role={action.get('role')} enabled={action.get('enabled')} "
                    f"name={action.get('name')!r} text={action.get('text')!r} "
                    f"id={action.get('id')}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
