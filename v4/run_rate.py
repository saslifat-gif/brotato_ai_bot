"""Run the unchanged V4 controller at one explicitly selected bridge rate."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from brotato_ai.training.configs import load_config
from v4.env.brotato_api_env import BrotatoApiEnv
from v4.combat_policy import HierarchicalCombatVectorizer
from brotato_ai.training.checkpoints import load_temporal_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen V4 controller rate-sensitivity run"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-hz", type=float, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PPO device. cpu uses the machine cores; cuda is slower for this MLP.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.environ.get("BROTATO_TORCH_THREADS", "8")),
        help="CPU threads for PPO/NumPy. Default 8 physical cores.",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--combat-dataset", type=Path)
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()
    if not 8.0 <= args.state_hz <= 60.0:
        parser.error("--state-hz must be between 8 and 60")
    if args.episodes < 0 or args.timesteps < 1:
        parser.error("episodes must be non-negative and timesteps must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be at least 1")

    os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.torch_threads)
    torch.set_num_threads(int(args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cfg = load_config()
    cfg = replace(
        cfg,
        safety_shield=not args.no_safety,
        combat_decision_log=(
            args.combat_dataset.resolve() if args.combat_dataset else None
        ),
    )
    raw_env = BrotatoApiEnv(
        cfg,
        vectorizer=HierarchicalCombatVectorizer(),
        state_hz=float(args.state_hz),
    )
    env = Monitor(raw_env)
    model = load_temporal_checkpoint(
        str(args.model.resolve()), env=env, device=args.device
    )
    print(
        f"[v4-frozen-rate] model={args.model.resolve()} state_hz={args.state_hz:g} "
        f"safety={cfg.safety_shield} device={args.device} "
        f"torch_threads={args.torch_threads} "
        f"logical_cpus={os.cpu_count()}",
        flush=True,
    )

    observation, info = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_id = 0
    action_changes = 0
    previous_action: int | None = None
    health_loss = 0.0
    projectile_hits = 0
    hazard_damage_events = 0
    safety_overrides = 0
    results: list[dict] = []
    try:
        for _ in range(int(args.timesteps)):
            inference_started = env.unwrapped.profiler.begin("model_predict")
            action, _ = model.predict(observation, deterministic=True)
            env.unwrapped.profiler.end("model_predict", inference_started)
            selected = int(np.asarray(action).reshape(-1)[0])
            if previous_action is not None:
                action_changes += int(selected != previous_action)
            previous_action = selected
            observation, reward, terminated, truncated, info = env.step(selected)
            episode_reward += float(reward)
            episode_steps += 1
            taken = float(info.get("damage_taken", 0.0) or 0.0)
            health_loss += max(0.0, taken)
            projectile_hits += int(float(info.get("damage_after_projectile_visible", 0.0) or 0.0) > 0.0)
            hazard_damage_events += int(float(info.get("damage_after_projectile_hazard", 0.0) or 0.0) > 0.0)
            safety_overrides += int(bool(info.get("safety_overridden", False)))
            if not (terminated or truncated):
                continue
            episode_id += 1
            results.append(
                {
                    "episode": episode_id,
                    "rate_hz": float(args.state_hz),
                    "model": str(args.model.resolve()),
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "wave": int(info.get("wave", 0)),
                    "dead": bool(info["dead"]),
                    "victory": bool(info["victory"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "health_fraction": float(info.get("health_fraction", 0.0) or 0.0),
                    "health_loss": health_loss,
                    "projectile_hits": projectile_hits,
                    "hazard_damage_events": hazard_damage_events,
                    "effective_state_hz": float(info.get("effective_state_hz", 0.0)),
                    "control_overruns": int(info.get("control_overruns", 0)),
                    "control_missed_ticks": int(info.get("control_missed_ticks", 0)),
                    "action_changes": action_changes,
                    "safety_overrides": safety_overrides,
                    "final_phase": info.get("phase"),
                }
            )
            args.results.parent.mkdir(parents=True, exist_ok=True)
            args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(
                f"[v4-frozen-rate] episode={episode_id} wave={results[-1]['wave']} "
                f"steps={episode_steps} dead={int(results[-1]['dead'])} "
                f"hp_loss={results[-1]['health_loss']:.1f} "
                f"proj_hits={results[-1]['projectile_hits']} "
                f"effective_hz={results[-1]['effective_state_hz']:.2f}",
                flush=True,
            )
            if args.episodes > 0 and episode_id >= args.episodes:
                break
            observation, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            action_changes = 0
            previous_action = None
            health_loss = 0.0
            projectile_hits = 0
            hazard_damage_events = 0
            safety_overrides = 0
    except KeyboardInterrupt:
        print("[v4-frozen-rate] stopped", flush=True)
        return 130
    finally:
        raw_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
