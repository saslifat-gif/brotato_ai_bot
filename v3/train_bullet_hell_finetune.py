"""Migrate full-arena PPO to all-projectile future-path PPO."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn
from torch import nn

from v3.combat_policy import (
    BULLET_HELL_OBSERVATION_SIZE,
    FULL_ARENA_OBSERVATION_SIZE,
    SEMANTIC_OBSERVATION_SIZE,
    BulletHellCombatVectorizer,
)
from v3.config import load_config
from v3.env.brotato_api_env import BrotatoApiEnv
from v3.runtime_callbacks import SaveBestRollingRewardCallback
from v3.train_combat_finetune import (
    CombatTensorboardCallback,
    HumanAnchoredPPO,
    actor_logits,
)
from v3.train_full_arena_finetune import FullArenaActorExtractor
from v3.train_semantic_combat_bc import load_semantic_records


class OfflineBulletHellEnv(gym.Env):
    action_space = spaces.Discrete(9)
    observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(BULLET_HELL_OBSERVATION_SIZE,),
        dtype=np.float32,
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(BULLET_HELL_OBSERVATION_SIZE, dtype=np.float32), {}

    def step(self, _action):
        raise RuntimeError("offline migration environment cannot be stepped")


class LegacyFullArenaPpoActor(nn.Module):
    """The complete trained full-arena extractor and its PPO action head."""

    actor_size = 9

    def __init__(self):
        super().__init__()
        shape = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FULL_ARENA_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.full_arena_extractor = FullArenaActorExtractor(shape)
        self.action_net = nn.Linear(
            self.actor_size + FULL_ARENA_OBSERVATION_SIZE,
            self.actor_size,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.full_arena_extractor(observations)
        return self.action_net(features)


class BulletHellActorExtractor(BaseFeaturesExtractor):
    """Preserve the full-arena actor and learn future-path residuals."""

    actor_size = 9

    def __init__(self, observation_space):
        super().__init__(
            observation_space,
            features_dim=self.actor_size + BULLET_HELL_OBSERVATION_SIZE,
        )
        self.legacy_actor = LegacyFullArenaPpoActor()
        new_feature_count = BULLET_HELL_OBSERVATION_SIZE - FULL_ARENA_OBSERVATION_SIZE
        self.bullet_residual = nn.Sequential(
            nn.LayerNorm(new_feature_count),
            nn.Linear(new_feature_count, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.actor_size),
        )
        nn.init.zeros_(self.bullet_residual[-1].weight)
        nn.init.zeros_(self.bullet_residual[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        old_observation = observations[..., :FULL_ARENA_OBSERVATION_SIZE]
        bullet_observation = observations[..., FULL_ARENA_OBSERVATION_SIZE:]
        logits = self.legacy_actor(old_observation)
        logits = logits + self.bullet_residual(bullet_observation)
        return torch.cat((logits, observations), dim=-1)


def initialize_bullet_hell_from_full_arena_ppo(
    model: HumanAnchoredPPO,
    source: HumanAnchoredPPO,
) -> float:
    """Copy the trained full-arena PPO actor with exact logit parity."""

    source_extractor = source.policy.pi_features_extractor
    target_extractor = model.policy.pi_features_extractor
    source_shape = tuple(source.observation_space.shape)
    source_action_inputs = int(getattr(source.policy.action_net, "in_features", -1))
    if (
        source_shape != (FULL_ARENA_OBSERVATION_SIZE,)
        or source_action_inputs != 9 + FULL_ARENA_OBSERVATION_SIZE
        or not hasattr(source_extractor, "full_arena_residual")
        or not hasattr(source_extractor, "legacy_actor")
    ):
        raise RuntimeError("source checkpoint is not a full-arena PPO policy")
    if not isinstance(target_extractor, BulletHellActorExtractor):
        raise RuntimeError("target policy is missing BulletHellActorExtractor")
    target_extractor.legacy_actor.full_arena_extractor.load_state_dict(
        source_extractor.state_dict()
    )
    target_extractor.legacy_actor.action_net.load_state_dict(
        source.policy.action_net.state_dict()
    )
    with torch.no_grad():
        # SB3 orthogonal initialization runs after custom extractor setup.
        target_extractor.bullet_residual[-1].weight.zero_()
        target_extractor.bullet_residual[-1].bias.zero_()
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[:, :9].copy_(
            torch.eye(9, device=model.device)
        )

    old_probe = torch.zeros((3, FULL_ARENA_OBSERVATION_SIZE), device=model.device)
    old_probe[1] = torch.linspace(
        -1.0, 1.0, FULL_ARENA_OBSERVATION_SIZE, device=model.device
    )
    old_probe[2] = torch.linspace(
        1.0, -1.0, FULL_ARENA_OBSERVATION_SIZE, device=model.device
    )
    new_probe = torch.zeros((3, BULLET_HELL_OBSERVATION_SIZE), device=model.device)
    new_probe[:, :FULL_ARENA_OBSERVATION_SIZE] = old_probe
    new_probe[:, FULL_ARENA_OBSERVATION_SIZE:] = torch.linspace(
        -1.0,
        1.0,
        BULLET_HELL_OBSERVATION_SIZE - FULL_ARENA_OBSERVATION_SIZE,
        device=model.device,
    )
    source.policy.to(model.device)
    with torch.no_grad():
        source_logits = actor_logits(source.policy, old_probe)
        target_logits = actor_logits(model.policy, new_probe)
        legacy_logits = target_extractor.legacy_actor(old_probe)
        residual = target_extractor.bullet_residual(
            new_probe[:, FULL_ARENA_OBSERVATION_SIZE:]
        )
        difference = float((target_logits - source_logits).abs().max().item())
        legacy_difference = float((legacy_logits - source_logits).abs().max().item())
        residual_magnitude = float(residual.abs().max().item())
    print(
        "[bullet-hell-ppo] transfer diagnostics "
        f"legacy_diff={legacy_difference:.8g} residual={residual_magnitude:.8g} "
        f"final_diff={difference:.8g}"
    )
    if difference > 1e-5:
        raise RuntimeError(f"bullet-hell actor transfer failed: max_abs_diff={difference}")
    return difference


def _padded_anchor_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(records), BULLET_HELL_OBSERVATION_SIZE), dtype=np.float32)
    semantic = np.asarray([record["features"] for record in records], dtype=np.float32)
    features[:, :SEMANTIC_OBSERVATION_SIZE] = semantic
    actions = np.asarray([int(record["action"]) for record in records], dtype=np.int64)
    return features, actions


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Migrate full-arena PPO to future projectile-path PPO"
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=cfg.output_dir / "full_arena_finetune_best" / "best_training_agent.zip",
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
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--safety", action="store_true")
    args = parser.parse_args()
    if not 4.0 <= args.state_hz <= 60.0:
        parser.error("--state-hz must be between 4 and 60")

    cfg = replace(cfg, safety_shield=bool(args.safety))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_semantic_records(args.dataset)
    if len(records) < 1_000:
        raise RuntimeError(f"only {len(records)} semantic records in {args.dataset}")
    features, actions = _padded_anchor_arrays(records)
    env = (
        OfflineBulletHellEnv()
        if args.bootstrap_only
        else Monitor(BrotatoApiEnv(
            cfg,
            vectorizer=BulletHellCombatVectorizer(),
            state_hz=args.state_hz,
        ))
    )
    checkpoints = cfg.output_dir / "bullet_hell_finetune_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    transfer_difference = None
    if args.resume:
        model = HumanAnchoredPPO.load(args.resume, env=env, device=args.device)
        model.learning_rate = max(1e-7, float(args.learning_rate))
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = model.learning_rate
        model.ent_coef = max(0.0, float(args.ent_coef))
        print(f"[bullet-hell-ppo] resumed={args.resume.resolve()}")
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
                "features_extractor_class": BulletHellActorExtractor,
                "net_arch": {"pi": [], "vf": [256, 128]},
                "activation_fn": nn.Tanh,
                "share_features_extractor": False,
            },
        )
        transfer_difference = initialize_bullet_hell_from_full_arena_ppo(model, source)
    model.bc_coefficient = max(0.0, float(args.bc_coefficient))
    model.bc_batches = max(0, int(args.bc_batches))
    model.set_human_anchor(features, actions)
    if not args.resume:
        bootstrap = cfg.output_dir / "bullet_hell_ppo_bootstrap"
        model.save(str(bootstrap))
        print(f"[bullet-hell-ppo] bootstrap model saved={bootstrap}.zip")
    print(
        f"[bullet-hell-ppo] source={args.source_model.resolve()} records={len(records)} "
        f"observation_size={BULLET_HELL_OBSERVATION_SIZE} "
        f"transfer_max_abs_diff={transfer_difference} state_hz={args.state_hz:g} "
        f"safety={cfg.safety_shield} bc_coefficient={model.bc_coefficient}"
    )
    if args.bootstrap_only:
        print("[bullet-hell-ppo] bootstrap-only migration verified; live bridge untouched")
        env.close()
        return 0
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=5_000,
            save_path=str(checkpoints),
            name_prefix="bullet_hell_ppo",
        ),
        SaveBestRollingRewardCallback(
            cfg.output_dir / "bullet_hell_finetune_best",
            min_episodes=10,
        ),
        CombatTensorboardCallback(),
    ])
    try:
        model.learn(
            total_timesteps=max(1, int(args.timesteps)),
            callback=callbacks,
            tb_log_name="BulletHellPPO",
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        target = cfg.output_dir / "bullet_hell_ppo_interrupted"
        model.save(str(target))
        print(f"[bullet-hell-ppo] interrupted model saved={target}.zip")
        return 130
    except Exception:
        target = cfg.output_dir / "bullet_hell_ppo_recovery"
        model.save(str(target))
        print(f"[bullet-hell-ppo] error recovery model saved={target}.zip")
        raise
    finally:
        env.close()
    target = cfg.output_dir / "bullet_hell_ppo_final"
    model.save(str(target))
    print(f"[bullet-hell-ppo] final model saved={target}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
