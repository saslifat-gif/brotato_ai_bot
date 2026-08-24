"""Run a short deterministic end-to-end v3 automation check without PPO."""

import argparse
import json

from v3.config import load_config
from v3.env.brotato_api_env import BrotatoApiEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test v3 combat and menu automation")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--segment", type=int, default=24)
    parser.add_argument(
        "--idle-after-wave",
        type=int,
        default=0,
        help="use idle actions from this wave onward to exercise death/retry",
    )
    parser.add_argument(
        "--stop-after-episodes",
        type=int,
        default=0,
        help="stop successfully after this many automatic episode resets",
    )
    parser.add_argument(
        "--semantic-details",
        action="store_true",
        help="print the first live sample of each semantic API entity",
    )
    args = parser.parse_args()
    env = BrotatoApiEnv(load_config())
    phases = []
    episodes = 0
    semantic_seen = set()
    try:
        _observation, info = env.reset()
        print(f"[v3-smoke] reset phase={info['phase']} tick={info['tick']}")
        start_wave = int(info.get("wave", 0))
        highest_wave = start_wave
        confirmed_ui = set(info.get("ui_confirmed", []))
        # Clockwise square: right, down, left, up.
        movement = (4, 2, 3, 1)
        completed_steps = 0
        for step in range(max(1, args.steps)):
            current_wave = int(info.get("wave", 0))
            should_idle = args.idle_after_wave > 0 and current_wave >= args.idle_after_wave
            action = 0 if should_idle else movement[(step // max(1, args.segment)) % len(movement)]
            _observation, reward, terminated, truncated, info = env.step(action)
            if args.semantic_details:
                state = env.last_state or {}
                combat = state.get("combat", {})
                semantic_sources = {
                    "enemy": state.get("enemies") or [],
                    "pickup": state.get("pickups") or [],
                    "weapon": combat.get("weapons") or [],
                    "attack_indicator": state.get("attack_indicators") or [],
                }
                for kind, entries in semantic_sources.items():
                    if kind not in semantic_seen and entries:
                        semantic_seen.add(kind)
                        print(
                            f"[v3-semantic] kind={kind} "
                            + json.dumps(entries[0], ensure_ascii=False)
                        )
            completed_steps = step + 1
            highest_wave = max(highest_wave, int(info.get("wave", 0)))
            confirmed_ui.update(info.get("ui_confirmed", []))
            phase = str(info.get("phase"))
            if not phases or phases[-1] != phase:
                phases.append(phase)
                state = env.last_state or {}
                print(
                    f"[v3-smoke] step={step + 1} phase={phase} "
                    f"wave={info.get('wave')} materials="
                    f"{state.get('counters', {}).get('materials')} reward={reward:.3f}"
                )
            elif (step + 1) % 100 == 0:
                state = env.last_state or {}
                print(
                    f"[v3-smoke] progress step={step + 1} wave={info.get('wave')} "
                    f"materials={state.get('counters', {}).get('materials')}"
                )
            if truncated:
                raise RuntimeError(f"automation stopped in non-combat phase={phase}")
            if terminated:
                episodes += 1
                _observation, info = env.reset()
                confirmed_ui.update(info.get("ui_confirmed", []))
                print(
                    f"[v3-smoke] episode_reset={episodes} "
                    f"phase={info['phase']} tick={info['tick']}"
                )
                if args.stop_after_episodes > 0 and episodes >= args.stop_after_episodes:
                    break
        if highest_wave > start_wave and "upgrade_choice" not in confirmed_ui:
            raise RuntimeError(
                "wave advanced without a bridge-confirmed upgrade selection; "
                "do not click the upgrade manually during this test"
            )
        print(
            f"[v3-smoke] ok steps={completed_steps} episodes={episodes} "
            f"phases={phases} confirmed_ui={sorted(confirmed_ui)}"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
