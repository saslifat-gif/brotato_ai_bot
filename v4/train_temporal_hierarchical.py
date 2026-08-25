"""Migrate V3 bullet PPO to a finite-history GRU hierarchical policy."""

from __future__ import annotations

import argparse
from datetime import datetime
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

from v3.combat_policy import BULLET_HELL_OBSERVATION_SIZE, SEMANTIC_OBSERVATION_SIZE
from v3.config import load_config
from v3.env.brotato_api_env import BrotatoApiEnv
from v3.runtime_callbacks import SaveBestRollingRewardCallback
from v3.train_bullet_hell_finetune import BulletHellActorExtractor
from v3.train_combat_finetune import CombatTensorboardCallback, HumanAnchoredPPO, actor_logits
from v3.train_semantic_combat_bc import load_semantic_records
from v4.combat_policy import (
    HISTORY_FEATURES,
    HISTORY_SIZE,
    HISTORY_STEPS,
    MACRO_FEATURES,
    V4_OBSERVATION_SIZE,
    HierarchicalCombatVectorizer,
)


OBJECTIVE_NAMES = ("evade", "heal", "loot", "engage", "reposition")


class OfflineV4Env(gym.Env):
    action_space = spaces.Discrete(9)
    observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(V4_OBSERVATION_SIZE,),
        dtype=np.float32,
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(V4_OBSERVATION_SIZE, dtype=np.float32), {}

    def step(self, _action):
        raise RuntimeError("offline V4 migration environment cannot be stepped")


class LegacyBulletPpoActor(nn.Module):
    actor_size = 9

    def __init__(self):
        super().__init__()
        shape = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(BULLET_HELL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.extractor = BulletHellActorExtractor(shape)
        self.action_net = nn.Linear(
            self.actor_size + BULLET_HELL_OBSERVATION_SIZE,
            self.actor_size,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.extractor(observations))


class TemporalHierarchicalActorExtractor(BaseFeaturesExtractor):
    """Old actor plus a zero-initialized GRU and macro residual."""

    actor_size = 9

    def __init__(self, observation_space):
        super().__init__(
            observation_space,
            features_dim=self.actor_size + V4_OBSERVATION_SIZE,
        )
        self.legacy_actor = LegacyBulletPpoActor()
        self.history_gru = nn.GRU(
            input_size=HISTORY_FEATURES,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
        )
        self.macro_encoder = nn.Sequential(
            nn.LayerNorm(MACRO_FEATURES),
            nn.Linear(MACRO_FEATURES, 32),
            nn.Tanh(),
        )
        self.temporal_residual = nn.Sequential(
            nn.Linear(96 + 32, 128),
            nn.Tanh(),
            nn.Linear(128, self.actor_size),
        )
        nn.init.zeros_(self.temporal_residual[-1].weight)
        nn.init.zeros_(self.temporal_residual[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        old = observations[..., :BULLET_HELL_OBSERVATION_SIZE]
        history_start = BULLET_HELL_OBSERVATION_SIZE
        history = observations[..., history_start:history_start + HISTORY_SIZE]
        history = history.reshape(-1, HISTORY_STEPS, HISTORY_FEATURES)
        macro = observations[..., history_start + HISTORY_SIZE:]
        _, hidden = self.history_gru(history)
        temporal = hidden[-1]
        residual = self.temporal_residual(
            torch.cat((temporal, self.macro_encoder(macro)), dim=-1)
        )
        logits = self.legacy_actor(old) + residual
        return torch.cat((logits, observations), dim=-1)


class V4TensorboardCallback(CombatTensorboardCallback):
    """Add inspectable macro planner and temporal-state curves."""

    def _on_step(self) -> bool:
        if not super()._on_step():
            return False
        observations = self.locals.get("new_obs")
        if observations is None:
            return True
        batch = np.asarray(observations, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.shape[-1] != V4_OBSERVATION_SIZE:
            return True
        macro_start = BULLET_HELL_OBSERVATION_SIZE + HISTORY_SIZE
        macro = batch[:, macro_start:]
        for index, name in enumerate(OBJECTIVE_NAMES):
            self.logger.record_mean(f"v4/objective_{name}", float(np.mean(macro[:, index])))
        self.logger.record_mean("v4/macro_target_x", float(np.mean(macro[:, -3])))
        self.logger.record_mean("v4/macro_target_y", float(np.mean(macro[:, -2])))
        self.logger.record_mean("v4/macro_urgency", float(np.mean(macro[:, -1])))
        history_start = BULLET_HELL_OBSERVATION_SIZE
        history = batch[:, history_start:history_start + HISTORY_SIZE]
        history = history.reshape(-1, HISTORY_STEPS, HISTORY_FEATURES)
        self.logger.record_mean(
            "v4/history_motion", float(np.mean(history[:, :, 11]))
        )
        self.logger.record_mean(
            "v4/history_health_delta", float(np.mean(history[:, :, 12]))
        )
        return True


def initialize_v4_from_bullet_ppo(
    model: HumanAnchoredPPO,
    source: HumanAnchoredPPO,
) -> float:
    """Copy the complete V3 actor and prove exact initial logit parity."""

    source_extractor = source.policy.pi_features_extractor
    target_extractor = model.policy.pi_features_extractor
    if (
        tuple(source.observation_space.shape) != (BULLET_HELL_OBSERVATION_SIZE,)
        or int(getattr(source.policy.action_net, "in_features", -1))
        != 9 + BULLET_HELL_OBSERVATION_SIZE
        or not isinstance(target_extractor, TemporalHierarchicalActorExtractor)
        or not hasattr(source_extractor, "bullet_residual")
        or not hasattr(source_extractor, "legacy_actor")
    ):
        raise RuntimeError("source checkpoint is not a V3 bullet-hell PPO policy")
    target_extractor.legacy_actor.extractor.load_state_dict(source_extractor.state_dict())
    target_extractor.legacy_actor.action_net.load_state_dict(
        source.policy.action_net.state_dict()
    )
    with torch.no_grad():
        # SB3 orthogonal initialization happens after extractor construction.
        target_extractor.temporal_residual[-1].weight.zero_()
        target_extractor.temporal_residual[-1].bias.zero_()
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[:, :9].copy_(torch.eye(9, device=model.device))

    old_probe = torch.zeros((4, BULLET_HELL_OBSERVATION_SIZE), device=model.device)
    old_probe[1] = torch.linspace(-1.0, 1.0, BULLET_HELL_OBSERVATION_SIZE, device=model.device)
    old_probe[2] = torch.linspace(1.0, -1.0, BULLET_HELL_OBSERVATION_SIZE, device=model.device)
    old_probe[3] = torch.rand(BULLET_HELL_OBSERVATION_SIZE, device=model.device) * 2.0 - 1.0
    new_probe = torch.rand((4, V4_OBSERVATION_SIZE), device=model.device) * 2.0 - 1.0
    new_probe[:, :BULLET_HELL_OBSERVATION_SIZE] = old_probe
    source.policy.to(model.device)
    with torch.no_grad():
        source_logits = actor_logits(source.policy, old_probe)
        target_logits = actor_logits(model.policy, new_probe)
        difference = float((source_logits - target_logits).abs().max().item())
    print(f"[v4] exact V3 transfer max_abs_diff={difference:.8g}")
    if difference > 1e-5:
        raise RuntimeError(f"V4 actor transfer failed: max_abs_diff={difference}")
    return difference


def balanced_anchor_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep representative IDLE demonstrations without letting them dominate."""

    idle = [record for record in records if int(record["action"]) == 0]
    moving = [record for record in records if int(record["action"]) != 0]
    idle_target = min(len(idle), max(1, int(len(moving) * 0.10 / 0.90)))
    if idle_target < len(idle):
        indices = np.linspace(0, len(idle) - 1, idle_target, dtype=np.int64)
        idle = [idle[int(index)] for index in indices]
    selected = moving + idle
    features = np.zeros((len(selected), V4_OBSERVATION_SIZE), dtype=np.float32)
    features[:, :SEMANTIC_OBSERVATION_SIZE] = np.asarray(
        [record["features"] for record in selected], dtype=np.float32
    )
    actions = np.asarray([int(record["action"]) for record in selected], dtype=np.int64)
    idle_fraction = float(np.mean(actions == 0))
    return features, actions, idle_fraction


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train V4 temporal hierarchical movement")
    parser.add_argument(
        "--source-model",
        type=Path,
        default=cfg.output_dir / "bullet_hell_finetune_best" / "best_training_agent.zip",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=cfg.output_dir / "human_semantic_combat_v2.jsonl",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--timesteps", type=int, default=cfg.total_timesteps)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--state-hz", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--ent-coef", type=float, default=0.0002)
    parser.add_argument("--bc-coefficient", type=float, default=0.10)
    parser.add_argument("--bc-batches", type=int, default=2)
    parser.add_argument(
        "--run-name",
        default=None,
        help="TensorBoard run name; defaults to a unique timestamp per launch",
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="disable the runtime safety shield (diagnostics only)",
    )
    parser.add_argument("--bootstrap-only", action="store_true")
    args = parser.parse_args()
    if not 8.0 <= args.state_hz <= 24.0:
        parser.error("--state-hz must be between 8 and 24")
    run_name = args.run_name or f"V4TemporalPPO_{datetime.now():%Y%m%d_%H%M%S}"

    # Keep the hard safety shield on for live training.  The old trainer
    # disabled it unconditionally, allowing the policy to walk into API-rated
    # enemy and boundary paths.  --no-safety remains available for diagnostics.
    cfg = replace(cfg, safety_shield=not args.no_safety)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_semantic_records(args.dataset)
    if len(records) < 1_000:
        raise RuntimeError(f"only {len(records)} semantic records in {args.dataset}")
    anchor_features, anchor_actions, idle_fraction = balanced_anchor_arrays(records)
    env = OfflineV4Env() if args.bootstrap_only else Monitor(BrotatoApiEnv(
        cfg,
        vectorizer=HierarchicalCombatVectorizer(),
        state_hz=args.state_hz,
    ))
    checkpoints = cfg.output_dir / "v4_temporal_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    difference = None
    if args.resume:
        model = HumanAnchoredPPO.load(args.resume, env=env, device=args.device)
        model.learning_rate = max(1e-7, float(args.learning_rate))
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = model.learning_rate
        model.ent_coef = max(0.0, float(args.ent_coef))
        print(f"[v4] resumed={args.resume.resolve()}")
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
                "features_extractor_class": TemporalHierarchicalActorExtractor,
                "net_arch": {"pi": [], "vf": [256, 128]},
                "activation_fn": nn.Tanh,
                "share_features_extractor": False,
            },
        )
        difference = initialize_v4_from_bullet_ppo(model, source)
    model.bc_coefficient = max(0.0, float(args.bc_coefficient))
    model.bc_batches = max(0, int(args.bc_batches))
    model.set_human_anchor(anchor_features, anchor_actions)
    bootstrap = cfg.output_dir / "v4_temporal_bootstrap"
    if not args.resume:
        model.save(str(bootstrap))
        print(f"[v4] bootstrap saved={bootstrap}.zip")
    print(
        f"[v4] observation={V4_OBSERVATION_SIZE} history={HISTORY_STEPS}x{HISTORY_FEATURES} "
        f"anchor_records={len(anchor_actions)} anchor_idle={idle_fraction:.3f} "
        f"transfer_diff={difference} state_hz={args.state_hz:g}"
    )
    if args.bootstrap_only:
        env.close()
        print("[v4] offline bootstrap verified; live V3 untouched")
        return 0

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=5_000,
            save_path=str(checkpoints),
            name_prefix="v4_temporal_ppo",
        ),
        SaveBestRollingRewardCallback(cfg.output_dir / "v4_temporal_best", min_episodes=10),
        V4TensorboardCallback(),
    ])
    try:
        model.learn(
            total_timesteps=max(1, int(args.timesteps)),
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        target = cfg.output_dir / "v4_temporal_interrupted"
        model.save(str(target))
        print(f"[v4] interrupted model saved={target}.zip")
        return 130
    except Exception:
        target = cfg.output_dir / "v4_temporal_recovery"
        model.save(str(target))
        print(f"[v4] error recovery model saved={target}.zip")
        raise
    finally:
        env.close()
    target = cfg.output_dir / "v4_temporal_final"
    model.save(str(target))
    print(f"[v4] final model saved={target}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
