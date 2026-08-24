"""Migrate the semantic PPO actor to a whole-arena observation and fine-tune it."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn
from torch import nn

from v3.combat_policy import (
    FULL_ARENA_OBSERVATION_SIZE,
    SEMANTIC_OBSERVATION_SIZE,
    FullArenaCombatVectorizer,
    SemanticCombatPolicyBase,
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


class LegacySemanticPpoActor(nn.Module):
    """The complete trained v2 actor, including its PPO action head."""

    actor_size = 9

    def __init__(self):
        super().__init__()
        self.semantic_actor = SemanticCombatPolicyBase()
        self.action_net = nn.Linear(
            self.actor_size + SEMANTIC_OBSERVATION_SIZE,
            self.actor_size,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        semantic_logits = self.semantic_actor(observations)
        features = torch.cat((semantic_logits, observations), dim=-1)
        return self.action_net(features)


class FullArenaActorExtractor(BaseFeaturesExtractor):
    """Preserve the v2 PPO actor and learn only a zero-initialized residual."""

    actor_size = 9

    def __init__(self, observation_space):
        super().__init__(
            observation_space,
            features_dim=self.actor_size + FULL_ARENA_OBSERVATION_SIZE,
        )
        self.legacy_actor = LegacySemanticPpoActor()
        new_feature_count = FULL_ARENA_OBSERVATION_SIZE - SEMANTIC_OBSERVATION_SIZE
        self.full_arena_residual = nn.Sequential(
            nn.LayerNorm(new_feature_count),
            nn.Linear(new_feature_count, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.actor_size),
        )
        nn.init.zeros_(self.full_arena_residual[-1].weight)
        nn.init.zeros_(self.full_arena_residual[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        old_observation = observations[..., :SEMANTIC_OBSERVATION_SIZE]
        arena_observation = observations[..., SEMANTIC_OBSERVATION_SIZE:]
        logits = self.legacy_actor(old_observation)
        logits = logits + self.full_arena_residual(arena_observation)
        return torch.cat((logits, observations), dim=-1)


def initialize_full_arena_from_semantic_ppo(
    model: HumanAnchoredPPO,
    source: HumanAnchoredPPO,
) -> float:
    """Copy a trained semantic PPO actor and verify exact action-logit parity."""

    from v3.train_semantic_finetune import SemanticActorExtractor

    source_extractor = source.policy.pi_features_extractor
    target_extractor = model.policy.pi_features_extractor
    if not isinstance(source_extractor, SemanticActorExtractor):
        raise RuntimeError("source checkpoint is not a semantic PPO policy")
    if not isinstance(target_extractor, FullArenaActorExtractor):
        raise RuntimeError("target policy is missing FullArenaActorExtractor")
    target_extractor.legacy_actor.semantic_actor.load_state_dict(
        source_extractor.semantic_actor.state_dict()
    )
    target_extractor.legacy_actor.action_net.load_state_dict(
        source.policy.action_net.state_dict()
    )
    with torch.no_grad():
        # SB3 applies its own orthogonal initialization after constructing a
        # custom extractor, so enforce the zero residual at migration time.
        target_extractor.full_arena_residual[-1].weight.zero_()
        target_extractor.full_arena_residual[-1].bias.zero_()
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[:, : FullArenaActorExtractor.actor_size].copy_(
            torch.eye(FullArenaActorExtractor.actor_size, device=model.device)
        )

    old_probe = torch.zeros((3, SEMANTIC_OBSERVATION_SIZE), device=model.device)
    old_probe[1] = torch.linspace(
        -1.0, 1.0, SEMANTIC_OBSERVATION_SIZE, device=model.device
    )
    old_probe[2] = torch.linspace(
        1.0, -1.0, SEMANTIC_OBSERVATION_SIZE, device=model.device
    )
    new_probe = torch.zeros((3, FULL_ARENA_OBSERVATION_SIZE), device=model.device)
    new_probe[:, :SEMANTIC_OBSERVATION_SIZE] = old_probe
    new_probe[:, SEMANTIC_OBSERVATION_SIZE:] = torch.linspace(
        -1.0,
        1.0,
        FULL_ARENA_OBSERVATION_SIZE - SEMANTIC_OBSERVATION_SIZE,
        device=model.device,
    )
    source.policy.to(model.device)
    with torch.no_grad():
        source_features = source_extractor(old_probe)
        source_latent = source.policy.mlp_extractor.forward_actor(source_features)
        source_logits = source.policy.action_net(source_latent)
        legacy_logits = target_extractor.legacy_actor(old_probe)
        residual_logits = target_extractor.full_arena_residual(
            new_probe[:, SEMANTIC_OBSERVATION_SIZE:]
        )
        target_features = target_extractor(new_probe)
        target_latent = model.policy.mlp_extractor.forward_actor(target_features)
        target_logits = model.policy.action_net(target_latent)
        difference = float(
            (target_logits - source_logits)
            .abs()
            .max()
            .item()
        )
        legacy_difference = float((legacy_logits - source_logits).abs().max().item())
        residual_magnitude = float(residual_logits.abs().max().item())
        actor_helper_difference = float(
            (actor_logits(model.policy, new_probe) - target_logits).abs().max().item()
        )
    print(
        "[full-arena-ppo] transfer diagnostics "
        f"legacy_diff={legacy_difference:.8g} residual={residual_magnitude:.8g} "
        f"helper_diff={actor_helper_difference:.8g} final_diff={difference:.8g}"
    )
    if difference > 1e-5:
        raise RuntimeError(f"full-arena actor transfer failed: max_abs_diff={difference}")
    return difference


def _padded_anchor_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(records), FULL_ARENA_OBSERVATION_SIZE), dtype=np.float32)
    semantic = np.asarray([record["features"] for record in records], dtype=np.float32)
    features[:, :SEMANTIC_OBSERVATION_SIZE] = semantic
    actions = np.asarray([int(record["action"]) for record in records], dtype=np.int64)
    return features, actions


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Migrate semantic PPO to whole-arena PPO and fine-tune"
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=cfg.output_dir / "semantic_finetune_best" / "best_training_agent.zip",
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

    cfg = replace(cfg, safety_shield=bool(args.safety))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_semantic_records(args.dataset)
    if len(records) < 1_000:
        raise RuntimeError(f"only {len(records)} semantic records in {args.dataset}")
    features, actions = _padded_anchor_arrays(records)
    env = Monitor(BrotatoApiEnv(
        cfg,
        vectorizer=FullArenaCombatVectorizer(),
        state_hz=args.state_hz,
    ))
    checkpoints = cfg.output_dir / "full_arena_finetune_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    transfer_difference = None
    if args.resume:
        model = HumanAnchoredPPO.load(args.resume, env=env, device=args.device)
        model.learning_rate = max(1e-7, float(args.learning_rate))
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = model.learning_rate
        model.ent_coef = max(0.0, float(args.ent_coef))
        print(f"[full-arena-ppo] resumed={args.resume.resolve()}")
    else:
        source = HumanAnchoredPPO.load(args.source_model, device=args.device)
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
                "features_extractor_class": FullArenaActorExtractor,
                "net_arch": {"pi": [], "vf": [256, 128]},
                "activation_fn": nn.Tanh,
                "share_features_extractor": False,
            },
        )
        transfer_difference = initialize_full_arena_from_semantic_ppo(model, source)
    model.bc_coefficient = max(0.0, float(args.bc_coefficient))
    model.bc_batches = max(0, int(args.bc_batches))
    model.set_human_anchor(features, actions)
    if not args.resume:
        bootstrap = cfg.output_dir / "full_arena_ppo_bootstrap"
        model.save(str(bootstrap))
        print(f"[full-arena-ppo] bootstrap model saved={bootstrap}.zip")
    print(
        f"[full-arena-ppo] source={args.source_model.resolve()} records={len(records)} "
        f"observation_size={FULL_ARENA_OBSERVATION_SIZE} "
        f"transfer_max_abs_diff={transfer_difference} state_hz={args.state_hz:g} "
        f"safety={cfg.safety_shield} bc_coefficient={model.bc_coefficient}"
    )
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=5_000,
            save_path=str(checkpoints),
            name_prefix="full_arena_ppo",
        ),
        SaveBestRollingRewardCallback(
            cfg.output_dir / "full_arena_finetune_best",
            min_episodes=10,
        ),
        CombatTensorboardCallback(),
    ])
    try:
        model.learn(
            total_timesteps=max(1, int(args.timesteps)),
            callback=callbacks,
            tb_log_name="FullArenaPPO",
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        target = cfg.output_dir / "full_arena_ppo_interrupted"
        model.save(str(target))
        print(f"[full-arena-ppo] interrupted model saved={target}.zip")
        return 130
    except Exception:
        target = cfg.output_dir / "full_arena_ppo_recovery"
        model.save(str(target))
        print(f"[full-arena-ppo] error recovery model saved={target}.zip")
        raise
    finally:
        env.close()
    target = cfg.output_dir / "full_arena_ppo_final"
    model.save(str(target))
    print(f"[full-arena-ppo] final model saved={target}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
