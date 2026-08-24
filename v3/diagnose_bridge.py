"""Receive and summarize live structured states from the Brotato mod."""

import argparse
import json

from v3.bridge_server import BridgeServer
from v3.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the v3 Brotato bridge")
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument(
        "--details",
        action="store_true",
        help="print one enemy, pickup, weapon and attack-indicator sample",
    )
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
            counters = state.get("counters", {})
            position = player.get("position", {})
            print(
                "[v3-state] "
                f"tick={previous_tick} phase={state.get('phase')} scene={state.get('scene')} "
                f"wave={wave.get('number')} hp={player.get('health')}/{player.get('max_health')} "
                f"pos=({position.get('x')},{position.get('y')}) "
                f"materials={counters.get('materials')} kills={counters.get('kills')} "
                f"enemies={len(state.get('enemies', []))} "
                f"projectiles={len(state.get('projectiles', []))} "
                f"pickups={len(state.get('pickups', []))} "
                f"indicators={len(state.get('attack_indicators', []))}"
            )
            if args.details:
                combat = state.get("combat", {})
                ui_actions = state.get("ui", {}).get("actions", [])
                samples = {
                    "enemy": (state.get("enemies") or [None])[0],
                    "pickup": (state.get("pickups") or [None])[0],
                    "weapon": (combat.get("weapons") or [None])[0],
                    "attack_indicator": (state.get("attack_indicators") or [None])[0],
                    "ui_actions": [
                        {
                            "role": action.get("role"),
                            "name": action.get("name"),
                            "text": action.get("text"),
                            "enabled": action.get("enabled"),
                        }
                        for action in ui_actions
                    ],
                }
                print("[v3-semantics] " + json.dumps(samples, ensure_ascii=False))
    print("[v3-diagnose] bridge state stream is healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
