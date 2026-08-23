"""Receive and summarize live structured states from the Brotato mod."""

import argparse

from v3.bridge_server import BridgeServer
from v3.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the v3 Brotato bridge")
    parser.add_argument("--states", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config()
    print("[v3-diagnose] start/restart Brotato with BrotatoRLBridge enabled")
    print("[v3-diagnose] then enter a wave; no screen capture is used")
    with BridgeServer(cfg.host, cfg.port) as server:
        previous_tick = None
        for _ in range(max(1, int(args.states))):
            state = server.wait_for_state(
                timeout_sec=cfg.reset_timeout_sec,
                after_tick=previous_tick,
            )
            previous_tick = int(state.get("tick", -1))
            player = state.get("player", {})
            wave = state.get("wave", {})
            print(
                "[v3-state] "
                f"tick={previous_tick} phase={state.get('phase')} scene={state.get('scene')} "
                f"wave={wave.get('number')} hp={player.get('health')}/{player.get('max_health')} "
                f"enemies={len(state.get('enemies', []))} "
                f"projectiles={len(state.get('projectiles', []))} "
                f"pickups={len(state.get('pickups', []))}"
            )
    print("[v3-diagnose] bridge state stream is healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
