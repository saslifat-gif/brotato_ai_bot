"""Fine-tune the API-semantic human combat base with anchored PPO."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from v3.combat_policy import (
    SEMANTIC_OBSERVATION_SIZE,
    SemanticCombatPolicyBase,
    SemanticCombatVectorizer,
    load_semantic_combat_base,
)
from v3.config import load_config
from v3.env.brotato_api_env import BrotatoApiEnv
from v3.runtime_callbacks import SaveBestRollingRewardCallback
from v3.train_combat_finetune import (
    CombatTensorboardCallback,
    HumanAnchoredPPO,
    actor_logits,
)
from v3.train_semantic_combat_bc import load_semantic_records


class SemanticActorExtractor(BaseFeaturesExtractor):
    """Expose exact BC logits to PPO while retaining all inputs for its critic."""

    actor_size = 9

    def __init__(self, observation_space):
        features_dim = self.actor_size + SEMANTIC_OBSERVATION_SIZE
        super().__init__(observation_space, features_dim=features_dim)
        self.semantic_actor = SemanticCombatPolicyBase()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        logits = self.semantic_actor(observations)
        return torch.cat((logits, observations), dim=-1)


def initialize_actor_from_semantic_base(
    model: HumanAnchoredPPO,
    base: SemanticCombatPolicyBase,
) -> float:
    """Make PPO's initial actor exactly equal the semantic BC actor."""

    extractor = model.policy.pi_features_extractor
    if not isinstance(extractor, SemanticActorExtractor):
        raise RuntimeError("PPO policy is missing SemanticActorExtractor")
    extractor.semantic_actor.load_state_dict(base.state_dict())
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[:, : SemanticActorExtractor.actor_size].copy_(
            torch.eye(SemanticActorExtractor.actor_size, device=model.device)
        )
    probe = torch.zeros((2, SEMANTIC_OBSERVATION_SIZE), device=model.device)
    probe[1] = torch.linspace(-1.0, 1.0, SEMANTIC_OBSERVATION_SIZE, device=model.device)
    base = base.to(model.device)
    with torch.no_grad():
        difference = float((actor_logits(model.policy, probe) - base(probe)).abs().max().item())
    if difference > 1e-5:
        raise RuntimeError(f"semantic actor transfer failed: max_abs_diff={difference}")
    return difference


def _anchor_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([record["features"] for record in records], dtype=np.float32)
    actions = np.asarray([int(record["action"]) for record in records], dtype=np.int64)
    return features, actions


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Fine-tune the Brotato semantic combat base with anchored PPO"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=cfg.output_dir / "semantic_combat_base_candidate.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=cfg.output_dir / "human_semantic_combat_v2.jsonl",
    )
    parser.add_argument("--timesteps", type=int, default=cfg.total_timesteps)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--state-hz", type=float, default=12.0)
    parser.add_argument("--bc-coefficient", type=float, default=0.20)
    parser.add_argument("--bc-batches", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--ent-coef", type=float, default=0.0002)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--safety", action="store_true")
    args = parser.parse_args()
    if not 4.0 <= args.state_hz <= 24.0:
        parser.error("--state-hz must be between 4 and 24")

    # A hidden action override would make PPO's on-policy update invalid.
    cfg = replace(cfg, safety_shield=bool(args.safety))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_semantic_records(args.dataset)
    if len(records) < 1_000:
        raise RuntimeError(f"only {len(records)} semantic records in {args.dataset}")
    features, actions = _anchor_arrays(records)
    env = Monitor(BrotatoApiEnv(
        cfg,
        vectorizer=SemanticCombatVectorizer(),
        state_hz=args.state_hz,
    ))
    checkpoints = cfg.output_dir / "semantic_finetune_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    base_metadata = {}
    transfer_difference = None
    if args.resume:
        model = HumanAnchoredPPO.load(args.resume, env=env, device=args.device)
        model.learning_rate = max(1e-7, float(args.learning_rate))
        model.lr_schedule = lambda _progress: model.learning_rate
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = model.learning_rate
        model.ent_coef = max(0.0, float(args.ent_coef))
        print(f"[semantic-ppo] resumed={args.resume.resolve()}")
    else:
        base, base_metadata = load_semantic_combat_base(args.base_model)
        model = HumanAnchoredPPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=max(1e-7, float(args.learning_rate)),
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=max(0.0, float(args.ent_coef)),
            tensorboard_log=str(cfg.output_dir / "logs"),
            device=args.device,
            bc_coefficient=args.bc_coefficient,
            bc_batches=args.bc_batches,
            policy_kwargs={
                "features_extractor_class": SemanticActorExtractor,
                "net_arch": {"pi": [], "vf": [128, 64]},
                "activation_fn": nn.Tanh,
                "share_features_extractor": False,
            },
        )
        transfer_difference = initialize_actor_from_semantic_base(model, base)
    model.bc_coefficient = max(0.0, float(args.bc_coefficient))
    model.bc_batches = max(0, int(args.bc_batches))
    model.set_human_anchor(features, actions)
    if not args.resume:
        bootstrap = cfg.output_dir / "semantic_base_ppo_bootstrap"
        model.save(str(bootstrap))
        print(f"[semantic-ppo] bootstrap model saved={bootstrap}.zip")
    print(
        f"[semantic-ppo] base={args.base_model.resolve()} records={len(records)} "
        f"validation_accuracy={base_metadata.get('validation_accuracy')} "
        f"transfer_max_abs_diff={transfer_difference} state_hz={args.state_hz:g} "
        f"safety={cfg.safety_shield} bc_coefficient={model.bc_coefficient} "
        f"learning_rate={model.learning_rate} ent_coef={model.ent_coef}"
    )
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=5_000,
            save_path=str(checkpoints),
            name_prefix="semantic_base_ppo",
        ),
        SaveBestRollingRewardCallback(
            cfg.output_dir / "semantic_finetune_best",
            min_episodes=10,
        ),
        CombatTensorboardCallback(),
    ])
    try:
        model.learn(
            total_timesteps=max(1, int(args.timesteps)),
            callback=callbacks,
            tb_log_name="SemanticBasePPO",
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        target = cfg.output_dir / "semantic_base_ppo_interrupted"
        model.save(str(target))
        print(f"[semantic-ppo] interrupted model saved={target}.zip")
        return 130
    except Exception:
        target = cfg.output_dir / "semantic_base_ppo_recovery"
        model.save(str(target))
        print(f"[semantic-ppo] error recovery model saved={target}.zip")
        raise
    finally:
        env.close()
    target = cfg.output_dir / "semantic_base_ppo_final"
    model.save(str(target))
    print(f"[semantic-ppo] final model saved={target}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
