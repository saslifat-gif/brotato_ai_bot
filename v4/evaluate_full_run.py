"""Frozen full-run evaluation with explicit provenance and outcome validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from brotato_ai.evaluation.full_run import FullRunMetrics, summarize, validate_start
from brotato_ai.policy.modes import PolicyMode
from brotato_ai.training.configs import load_config
from v4.combat_policy import HierarchicalCombatVectorizer
from v4.env.brotato_api_env import BrotatoApiEnv
from brotato_ai.training.checkpoints import load_temporal_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--human-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--state-hz", type=float, default=24)
    parser.add_argument("--character", default="character_well_rounded")
    parser.add_argument("--weapon", default="weapon_smg")
    parser.add_argument("--difficulty", type=int, help="Expected difficulty; requires bridge difficulty telemetry")
    parser.add_argument("--variant", choices=("baseline", "shop", "movement"), default="baseline")
    parser.add_argument("--operator-notes", default="", help="Record manually confirmed settings that the bridge cannot verify")
    parser.add_argument("--episode-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.episodes < 1 or not 4 <= args.state_hz <= 60 or args.episode_seconds <= 0:
        parser.error("episodes and timeout must be positive; state-hz must be 4..60")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ["BROTATO_V4_FULL_RESTART"] = "1"
    cfg = replace(load_config(), policy_mode=PolicyMode.SHADOW_HUMAN,
                  human_model_path=args.human_model.resolve(), ui_model_path=None,
                  automate_menus=True, safety_shield=True, ui_build_profile="ranged_smg",
                  ui_decision_log=args.output / "shop.jsonl",
                  combat_decision_log=args.output / "decisions.jsonl",
                  reset_timeout_sec=90, state_timeout_sec=15, control_hz=args.state_hz)
    cfg.validate()
    model = load_temporal_checkpoint(str(args.model.resolve()), device="cpu")
    if model.observation_space.shape != (HierarchicalCombatVectorizer.observation_size,):
        raise ValueError("Checkpoint does not match the V4 hierarchical observation space")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "diff", "--name-only"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "commit": commit, "source_modified": dirty,
              "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
              "human_model_sha256": sha256(args.human_model),
              "policy_mode": "SHADOW_HUMAN", "build_profile": "ranged_smg",
              "requested_hz": args.state_hz, "full_restart": True,
              "expected_character": args.character, "expected_weapon": args.weapon,
              "expected_difficulty": args.difficulty, "seed_control": "not exposed by bridge",
              "variant": args.variant, "operator_notes": args.operator_notes,
              "status": "running", "runs": []}
    result_path = args.output / "results.json"
    save(result_path, report)
    env = None
    try:
        env = BrotatoApiEnv(cfg, vectorizer=HierarchicalCombatVectorizer(), state_hz=args.state_hz)
        if env.policy_mode is not PolicyMode.SHADOW_HUMAN:
            raise RuntimeError("Shadow model failed to load; refusing a silently different experiment")
        planner = None
        if args.variant == "shop":
            from v4.experimental_shop import AdaptiveSmgTeacher
            env.ui_controller._teacher = AdaptiveSmgTeacher()
        elif args.variant == "movement":
            from brotato_ai.control.route_planner import EscapeRoutePlanner
            planner = EscapeRoutePlanner()
        for episode in range(1, args.episodes + 1):
            if episode == 1:
                # Do not buy items or advance a resumed shop before discovering
                # that the resulting episode cannot count as a full run.
                preflight = env.server.wait_for_state(timeout_sec=cfg.reset_timeout_sec)
                if preflight.get("phase") != "game_over":
                    validate_start(preflight, character=args.character, weapon=args.weapon)
            observation, info = env.reset()
            start = validate_start(env.last_state, character=args.character, weapon=args.weapon)
            if args.difficulty is not None and start["difficulty"] != args.difficulty:
                raise ValueError(f"Difficulty not verified: expected {args.difficulty}, got {start['difficulty']}")
            report["bridge_hello"] = env.server.last_hello
            metrics, started = FullRunMetrics(), time.monotonic()
            terminated = truncated = False
            print(f"[full-run] episode={episode} start={start}", flush=True)
            last_wave = 0
            while time.monotonic() - started < args.episode_seconds:
                action, _ = model.predict(observation, deterministic=True)
                selected = int(np.asarray(action).item())
                if planner is not None:
                    selected, proposal = planner.propose(env.last_state, selected)
                    with (args.output / "route_proposals.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"tick": env.last_state["tick"], **proposal}) + "\n")
                observation, reward, terminated, truncated, info = env.step(selected)
                metrics.observe(reward, info, env.last_state)
                if metrics.max_wave != last_wave:
                    last_wave = metrics.max_wave
                    print(f"[full-run] episode={episode} wave={last_wave} hp={info['health_fraction']:.3f}", flush=True)
                if terminated or truncated:
                    break
            else:
                truncated = True
            result = metrics.finish(info, terminated=terminated, truncated=truncated, elapsed=time.monotonic()-started)
            result.update(episode=episode, start=start)
            report["runs"].append(result)
            report["summary"] = summarize(report["runs"])
            save(result_path, report)
            print(f"[full-run] episode={episode} outcome={result['outcome']} wave={result['max_wave']}", flush=True)
            if not result["valid_full_run"]:
                raise RuntimeError("Incomplete run: stopped rather than counting a timeout or menu interruption as a death")
        report["status"] = "complete"
        return 0
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            env.close()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(result_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
