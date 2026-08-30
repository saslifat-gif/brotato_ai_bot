"""Frame-by-frame terminal replay/debugger for human demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brotato_ai.data.human_demo import replay_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one human demonstration frame")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    frame = replay_frame(args.dataset, args.frame_id)
    if args.pretty:
        print(json.dumps(frame, indent=2, sort_keys=True))
    else:
        controller = frame["controller"]
        derived = frame["derived"]
        print(
            f"frame={frame['frame_id']} episode={frame['episode_id']} "
            f"phase={frame['phase']} wave={frame['wave']} action={frame['action']} "
            f"safest={controller.get('safest_action')} "
            f"human_input={frame['input'].get('source')} "
            f"raw_available={frame['input'].get('raw_available')} "
            f"nearest_enemy={derived.get('nearest_enemy_distance')} "
            f"projectile_tti_ms={derived.get('nearest_projectile_tti_ms')} "
            f"escape={derived.get('escape')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
