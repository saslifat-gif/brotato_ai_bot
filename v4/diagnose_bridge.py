"""Receive and summarize live structured states from the Brotato mod."""

import argparse
import json

from v4.bridge_server import BridgeServer
from v4.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the v4 Brotato bridge")
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument(
        "--details",
        action="store_true",
        help="print one enemy, pickup, weapon and attack-indicator sample",
    )
    args = parser.parse_args()
    cfg = load_config()
    print("[v4-diagnose] start/restart Brotato with BrotatoRLBridge enabled")
    print("[v4-diagnose] then enter a wave; no screen capture is used")
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
            paths = state.get("projectile_paths", {})
            path_risks = paths.get("action_risk", []) if isinstance(paths, dict) else []
            enemy_risks = (
                paths.get("enemy_action_risk", []) if isinstance(paths, dict) else []
            )
            boundary_risks = (
                paths.get("boundary_action_risk", []) if isinstance(paths, dict) else []
            )
            position = player.get("position", {})
            print(
                "[v4-state] "
                f"tick={previous_tick} phase={state.get('phase')} scene={state.get('scene')} "
                f"wave={wave.get('number')} hp={player.get('health')}/{player.get('max_health')} "
                f"pos=({position.get('x')},{position.get('y')}) "
                f"materials={counters.get('materials')} kills={counters.get('kills')} "
                f"enemies={len(state.get('enemies', []))} "
                f"projectiles={len(state.get('projectiles', []))} "
                f"hostile_paths={paths.get('count', 0) if isinstance(paths, dict) else 0} "
                f"path_risk={max(path_risks, default=0.0):.3f} "
                f"contact_risk={max(enemy_risks, default=0.0):.3f} "
                f"boundary_risk={max(boundary_risks, default=0.0):.3f} "
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
                    "projectile_paths": {
                        "columns": paths.get("columns"),
                        "rows": paths.get("rows"),
                        "channels": paths.get("channels"),
                        "hostile_count": paths.get("count"),
                        "enemy_count": paths.get("enemy_count"),
                        "active_grid_values": sum(
                            1 for value in paths.get("grid", []) if float(value) > 0.0
                        ),
                        "action_risk": path_risks,
                        "enemy_action_risk": enemy_risks,
                        "boundary_action_risk": boundary_risks,
                    },
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
                print("[v4-semantics] " + json.dumps(samples, ensure_ascii=False))
    print("[v4-diagnose] bridge state stream is healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
