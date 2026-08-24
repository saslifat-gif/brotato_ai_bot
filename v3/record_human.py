"""Record human combat movement while the structured teacher operates menus."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from v3.bridge_server import BridgeServer
from v3.combat_policy import SemanticHumanCombatDecisionLogger
from v3.config import load_config
from v3.protocol import MoveAction
from v3.ui_automation import AutoUiController


HUMAN_INPUT_CAPABILITY = "human_input_observation"
SEMANTIC_CAPABILITY = "semantic_entities_v2"


def require_human_input_capability(hello: object) -> None:
    payload = hello if isinstance(hello, dict) else {}
    capabilities = payload.get("capabilities", [])
    if (
        HUMAN_INPUT_CAPABILITY not in capabilities
        or SEMANTIC_CAPABILITY not in capabilities
    ):
        raise RuntimeError(
            "Bridge 0.3.1+ with human input and semantic entities is required; "
            "reinstall the v3 mod and restart Brotato"
        )


def should_record(
    action: int,
    previous_observed_action: int,
    elapsed_sec: float,
    *,
    sample_hz: float,
    idle_hz: float,
) -> bool:
    if int(action) != int(previous_observed_action):
        return True
    frequency = idle_hz if int(action) == int(MoveAction.IDLE) else sample_hz
    return elapsed_sec >= 1.0 / max(0.1, float(frequency))


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Record structured human WASD demonstrations; AI controls menus"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.output_dir / "human_semantic_combat_v2.jsonl",
    )
    parser.add_argument("--sample-hz", type=float, default=8.0)
    parser.add_argument("--idle-hz", type=float, default=2.0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    if args.sample_hz <= 0 or args.idle_hz <= 0:
        parser.error("sample frequencies must be positive")

    output = args.output.resolve()
    logger = SemanticHumanCombatDecisionLogger(output)
    server = BridgeServer(cfg.host, cfg.port)
    controller = AutoUiController(
        max_shop_buys=cfg.max_shop_buys,
        max_shop_rerolls=cfg.max_shop_rerolls,
        build_profile=cfg.ui_build_profile,
        ui_model_path=cfg.ui_model_path,
        decision_log_path=cfg.ui_decision_log,
    )
    server.start()
    sequence = 0
    episode = 0
    records = 0
    previous_action = int(MoveAction.IDLE)
    previous_observed_action = int(MoveAction.IDLE)
    last_recorded = 0.0
    last_phase = ""
    verified_session = ""
    state = None
    print(
        f"[v3-human] output={output} combat_control=human ui_control=ai "
        f"sample_hz={args.sample_hz:g} idle_hz={args.idle_hz:g}"
    )
    print("[v3-human] play with WASD; no combat actions will be sent by Python")

    try:
        while args.max_records <= 0 or records < args.max_records:
            previous_tick = int(state.get("tick", -1)) if state else None
            state = server.wait_for_state(
                timeout_sec=cfg.reset_timeout_sec,
                after_tick=previous_tick,
            )
            session = str(state.get("session", ""))
            if session != verified_session:
                require_human_input_capability(server.last_hello)
                verified_session = session
                print(f"[v3-human] verified_bridge_session={session}")
            phase = str(state.get("phase", "menu"))
            if phase != "combat":
                if last_phase == "combat" and phase in {"game_over", "victory"}:
                    episode += 1
                    controller.reset_episode()
                    print(f"[v3-human] episode_complete={episode} records={records}")
                previous_action = int(MoveAction.IDLE)
                previous_observed_action = int(MoveAction.IDLE)
                last_recorded = 0.0
                last_phase = phase
                if cfg.automate_menus and phase != "victory":
                    result = controller.advance(
                        server,
                        state,
                        sequence,
                        cfg.reset_timeout_sec,
                        allow_restart=True,
                    )
                    state = result.state
                    sequence = result.sequence
                    last_phase = str(state.get("phase", phase))
                continue

            action = int(state.get("human_action", MoveAction.IDLE))
            try:
                action = int(MoveAction(action))
            except ValueError:
                action = int(MoveAction.IDLE)
            now = time.monotonic()
            if should_record(
                action,
                previous_observed_action,
                now - last_recorded,
                sample_hz=args.sample_hz,
                idle_hz=args.idle_hz,
            ):
                logger.record(
                    state,
                    action,
                    previous_action=previous_action,
                    episode=episode,
                )
                previous_action = action
                last_recorded = now
                records += 1
                if records % 250 == 0:
                    wave = int(state.get("wave", {}).get("number", 0))
                    print(
                        f"[v3-human] records={records} episode={episode} "
                        f"wave={wave} action={action}"
                    )
            previous_observed_action = action
            last_phase = phase
    except KeyboardInterrupt:
        print(f"[v3-human] stopped records={records} output={output}")
        return 130
    finally:
        server.close()
    print(f"[v3-human] complete records={records} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
