"""Run a frozen deterministic combat policy while automating and logging UI."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from v3.combat_policy import (
    CombatHeuristicTeacher,
    CombatPolicyBase,
    RichCombatVectorizer,
    SemanticCombatVectorizer,
    load_semantic_combat_base,
    load_combat_base,
)
from v3.config import load_config


def load_combat_bc(path: Path) -> tuple[CombatPolicyBase, dict]:
    """Backward-compatible alias used by existing evaluation commands."""

    return load_combat_base(path)


def main() -> int:
    from v3.env.brotato_api_env import BrotatoApiEnv

    parser = argparse.ArgumentParser(
        description="Frozen Brotato policy runner for evaluation and safe data collection"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--policy",
        choices=("model", "bc", "semantic", "teacher"),
        default="model",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--combat-dataset", type=Path)
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()
    if args.policy in {"model", "bc", "semantic"} and args.model is None:
        parser.error(f"--model is required with --policy {args.policy}")
    if args.policy == "model" and RecurrentPPO is None:
        raise RuntimeError("sb3-contrib is required: pip install sb3-contrib")

    cfg = load_config()
    cfg = replace(
        cfg,
        safety_shield=not args.no_safety,
        combat_decision_log=args.combat_dataset.resolve() if args.combat_dataset else None,
    )
    env = BrotatoApiEnv(cfg)
    model = None
    bc_model = None
    bc_vectorizer = None
    semantic_model = None
    semantic_vectorizer = None
    teacher = None
    source = args.policy
    if args.policy == "model":
        model = RecurrentPPO.load(str(args.model.resolve()), device="auto")
        print(f"[v3-frozen] model={args.model.resolve()} deterministic=True")
    elif args.policy == "bc":
        bc_model, metadata = load_combat_bc(args.model)
        bc_vectorizer = RichCombatVectorizer()
        print(
            f"[v3-frozen] combat_bc={args.model.resolve()} deterministic=True "
            f"validation_accuracy={metadata.get('validation_accuracy')} "
            f"best_epoch={metadata.get('best_epoch')}"
        )
    elif args.policy == "semantic":
        semantic_model, metadata = load_semantic_combat_base(args.model)
        semantic_vectorizer = SemanticCombatVectorizer()
        print(
            f"[v3-frozen] semantic_base={args.model.resolve()} deterministic=True "
            f"validation_accuracy={metadata.get('validation_accuracy')} "
            f"best_epoch={metadata.get('best_epoch')}"
        )
    else:
        teacher = CombatHeuristicTeacher()
        print("[v3-frozen] policy=structured_teacher")
    print(
        f"[v3-frozen] safety={cfg.safety_shield} ui_dataset={cfg.ui_decision_log} "
        f"combat_dataset={cfg.combat_decision_log}"
    )

    observation, info = env.reset()
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)
    episode_reward = 0.0
    episode_steps = 0
    episode_shield_overrides = 0
    episode_action_counts = [0] * 9
    completed = 0
    results = []
    try:
        for _step in range(max(1, int(args.timesteps))):
            if model is not None:
                action, recurrent_state = model.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                selected = int(np.asarray(action).reshape(-1)[0])
            elif bc_model is not None:
                rich = bc_vectorizer.build(env.last_state or {}, env.previous_action)
                with torch.no_grad():
                    selected = int(
                        bc_model(torch.from_numpy(rich).unsqueeze(0)).argmax(dim=1).item()
                    )
            elif semantic_model is not None:
                semantic = semantic_vectorizer.build(env.last_state or {}, env.previous_action)
                with torch.no_grad():
                    selected = int(
                        semantic_model(torch.from_numpy(semantic).unsqueeze(0))
                        .argmax(dim=1)
                        .item()
                    )
            else:
                selected = int(teacher.select(env.last_state or {}))
            observation, reward, terminated, truncated, info = env.step(selected)
            episode_reward += float(reward)
            episode_steps += 1
            episode_action_counts[selected] += 1
            episode_shield_overrides += int(bool(info.get("safety_overridden")))
            done = bool(terminated or truncated)
            episode_start = np.asarray([done], dtype=bool)
            if not done:
                continue
            summary = {
                "episode": completed + 1,
                "reward": episode_reward,
                "steps": episode_steps,
                "wave": int(info.get("wave", 0)),
                "policy": source,
                "safety": cfg.safety_shield,
                "shield_overrides": episode_shield_overrides,
                "shield_rate": episode_shield_overrides / max(1, episode_steps),
                "requested_action_counts": episode_action_counts,
            }
            results.append(summary)
            completed += 1
            print(
                f"[v3-frozen] episode={completed} reward={episode_reward:.3f} "
                f"steps={episode_steps} wave={summary['wave']}"
            )
            if args.results:
                args.results.parent.mkdir(parents=True, exist_ok=True)
                args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
            if args.episodes > 0 and completed >= args.episodes:
                break
            observation, info = env.reset()
            recurrent_state = None
            episode_start[:] = True
            episode_reward = 0.0
            episode_steps = 0
            episode_shield_overrides = 0
            episode_action_counts = [0] * 9
    except KeyboardInterrupt:
        print("[v3-frozen] stopped")
        return 130
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
