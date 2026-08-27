"""Fine-tune the compact human combat base with PPO and a BC anchor."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F

from v3.combat_policy import (
    CombatPolicyBase,
    RICH_OBSERVATION_SIZE,
    RichCombatVectorizer,
    load_combat_base,
)
from v3.config import load_config
from v3.env.brotato_api_env import BrotatoApiEnv
from v3.runtime_callbacks import SaveBestRollingRewardCallback
from v3.train_combat_bc import load_records


class CombatLayerNormExtractor(BaseFeaturesExtractor):
    """The exact input normalization layer used by CombatPolicyBase."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=RICH_OBSERVATION_SIZE)
        self.layer_norm = nn.LayerNorm(RICH_OBSERVATION_SIZE)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(observations)


def actor_logits(policy, observations: torch.Tensor) -> torch.Tensor:
    policy_device = next(policy.parameters()).device
    observations = observations.to(policy_device)
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    latent = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent)


def initialize_actor_from_human_base(model: PPO, base: CombatPolicyBase) -> float:
    """Copy every human-base actor weight into the equivalent PPO actor."""

    extractor = model.policy.features_extractor
    if not isinstance(extractor, CombatLayerNormExtractor):
        raise RuntimeError("PPO policy is missing CombatLayerNormExtractor")
    extractor.layer_norm.load_state_dict(base.network[0].state_dict())
    policy_linears = [
        layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, nn.Linear)
    ]
    base_linears = [base.network[1], base.network[3]]
    if len(policy_linears) != len(base_linears):
        raise RuntimeError("PPO actor layout does not match the human combat base")
    for target, source in zip(policy_linears, base_linears):
        target.load_state_dict(source.state_dict())
    model.policy.action_net.load_state_dict(base.network[5].state_dict())

    probe = torch.zeros((2, RICH_OBSERVATION_SIZE), device=model.device)
    probe[1] = torch.linspace(-1.0, 1.0, RICH_OBSERVATION_SIZE, device=model.device)
    base = base.to(model.device)
    with torch.no_grad():
        difference = float((actor_logits(model.policy, probe) - base(probe)).abs().max().item())
    if difference > 1e-5:
        raise RuntimeError(f"human actor transfer verification failed: max_abs_diff={difference}")
    return difference


class HumanAnchoredPPO(PPO):
    """PPO with a small supervised update after each rollout to limit forgetting."""

    def __init__(
        self,
        *args,
        bc_coefficient: float = 0.20,
        bc_batches: int = 2,
        bc_batch_size: int = 256,
        **kwargs,
    ):
        self.bc_coefficient = float(bc_coefficient)
        self.bc_batches = max(0, int(bc_batches))
        self.bc_batch_size = max(16, int(bc_batch_size))
        self._bc_features = None
        self._bc_actions = None
        self._bc_validation_features = None
        self._bc_validation_actions = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_bc_features", "_bc_actions",
            "_bc_validation_features", "_bc_validation_actions",
        ]

    def set_human_anchor(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        *,
        validation_fraction: float = 0.10,
        seed: int = 17,
    ) -> None:
        if len(features) != len(actions) or len(features) == 0:
            raise ValueError("human anchor requires matching non-empty features and actions")
        fraction = min(0.5, max(0.0, float(validation_fraction)))
        rng = np.random.default_rng(int(seed))
        indices = rng.permutation(len(features))
        validation_count = min(len(features) - 1, max(1, int(len(features) * fraction)))
        validation_indices = indices[:validation_count]
        training_indices = indices[validation_count:]
        self._bc_features = torch.as_tensor(
            features[training_indices], dtype=torch.float32, device=self.device
        )
        self._bc_actions = torch.as_tensor(
            actions[training_indices], dtype=torch.long, device=self.device
        )
        self._bc_validation_features = torch.as_tensor(
            features[validation_indices], dtype=torch.float32, device=self.device
        )
        self._bc_validation_actions = torch.as_tensor(
            actions[validation_indices], dtype=torch.long, device=self.device
        )

    def train(self) -> None:
        """Pause the live simulator so gradient updates cannot cause deaths."""

        env = self.get_env()
        paused = False
        if env is not None:
            try:
                env.env_method("set_training_paused", True)
                paused = True
            except AttributeError:
                # Offline/bootstrap-only environments have no live game.
                pass
        try:
            self._train_while_game_paused()
        finally:
            if paused:
                env.env_method("set_training_paused", False)

    def _train_while_game_paused(self) -> None:
        super().train()
        if self._bc_features is None or self.bc_batches <= 0 or self.bc_coefficient <= 0.0:
            return
        self.policy.set_training_mode(True)
        losses = []
        accuracies = []
        count = int(self._bc_features.shape[0])
        for _ in range(self.bc_batches):
            indices = torch.randint(count, (min(self.bc_batch_size, count),), device=self.device)
            observations = self._bc_features[indices]
            targets = self._bc_actions[indices]
            logits = actor_logits(self.policy, observations)
            loss = F.cross_entropy(logits, targets)
            self.policy.optimizer.zero_grad()
            (loss * self.bc_coefficient).backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            accuracies.append(float((logits.argmax(dim=1) == targets).float().mean().item()))
        self.logger.record("human_bc/cross_entropy", float(np.mean(losses)))
        self.logger.record("human_bc/accuracy", float(np.mean(accuracies)))
        if self._bc_validation_features is not None:
            with torch.no_grad():
                validation_logits = actor_logits(
                    self.policy, self._bc_validation_features
                )
                validation_targets = self._bc_validation_actions
                validation_loss = F.cross_entropy(validation_logits, validation_targets)
                validation_prediction = validation_logits.argmax(dim=1)
                validation_accuracy = (
                    validation_prediction == validation_targets
                ).float().mean()
                self.logger.record(
                    "human_bc/validation_cross_entropy",
                    float(validation_loss.cpu().item()),
                )
                self.logger.record(
                    "human_bc/validation_accuracy",
                    float(validation_accuracy.cpu().item()),
                )
                for action in range(9):
                    mask = validation_targets == action
                    if bool(mask.any()):
                        self.logger.record(
                            f"human_bc/validation_accuracy_action_{action}",
                            float(
                                (validation_prediction[mask] == action)
                                .float().mean().cpu().item()
                            ),
                        )
        self.logger.record("human_bc/coefficient", self.bc_coefficient)


class CombatTensorboardCallback(BaseCallback):
    """Expose game-specific progress that SB3 cannot infer from reward alone."""

    def __init__(self):
        super().__init__()
        self.best_wave = 0
        self.deaths_by_wave: dict[int, int] = {}
        self.total_deaths = 0
        self.episode_id = 1
        self.total_victories = 0
        self.total_wave_clears = 0
        self.conditional_counts: dict[str, int] = {}
        self.conditional_sums: dict[str, float] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for index, info in enumerate(infos):
            wave = int(info.get("wave", 0))
            applied = int(info.get("applied_action", 0))
            requested = int(info.get("requested_action", applied))
            self.best_wave = max(self.best_wave, wave)
            self.logger.record_mean("combat/victory", float(bool(info.get("victory"))))
            self.logger.record_mean("combat/wave_clear", float(bool(info.get("wave_clear"))))
            self.logger.record_mean(
                "actions/requested_applied_disagreement",
                float(requested != applied),
            )
            for action in range(9):
                self.logger.record_mean(
                    f"actions/requested_{action}", float(requested == action)
                )
            self.logger.record_mean(
                "combat/min_action_risk",
                float(info.get("minimum_action_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/unsafe_action_count",
                float(info.get("unsafe_action_count", 0.0)),
            )
            self.logger.record_mean(
                "combat/unsafe_action_fraction",
                float(info.get("unsafe_action_fraction", 0.0)),
            )
            self.logger.record_mean(
                "combat/requested_to_minimum_regret",
                float(info.get("requested_to_minimum_regret", 0.0)),
            )
            self.logger.record_mean("combat/current_wave", wave)
            self.logger.record("combat/best_wave", self.best_wave)
            self.logger.record("combat/episode_id", self.episode_id)
            self.logger.record_mean(
                "combat/safety_override_rate", float(bool(info.get("safety_overridden")))
            )
            self.logger.record_mean(
                "movement/enemy_contact_override_rate",
                float(bool(info.get("enemy_contact_overridden"))),
            )
            self.logger.record_mean(
                "movement/enemy_contact_requested_risk",
                float(info.get("enemy_contact_requested_risk", 0.0)),
            )
            self.logger.record_mean(
                "movement/enemy_contact_applied_risk",
                float(info.get("enemy_contact_applied_risk", 0.0)),
            )
            self.logger.record_mean(
                "movement/enemy_contact_override_penalty",
                float(info.get("enemy_contact_override_penalty", 0.0)),
            )
            self.logger.record_mean(
                "movement/crowd_recovery_override_rate",
                float(bool(info.get("crowd_recovery_overridden"))),
            )
            self.logger.record_mean("combat/health_fraction", float(info.get("health_fraction", 0.0)))
            self.logger.record_mean("combat/materials", float(info.get("materials", 0)))
            self.logger.record_mean("combat/enemy_count", float(info.get("enemy_count", 0)))
            self.logger.record_mean("combat/projectile_count", float(info.get("projectile_count", 0)))
            self.logger.record_mean(
                "combat/attack_indicator_count",
                float(info.get("attack_indicator_count", 0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_count",
                float(info.get("projectile_path_count", 0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_max_risk",
                float(info.get("projectile_path_max_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_action_risk",
                float(info.get("projectile_path_action_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_visible_before_action",
                float(bool(info.get("projectile_visible_before_action"))),
            )
            self.logger.record_mean(
                "combat/projectile_count_before_action",
                float(info.get("projectile_count_before_action", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_total_count_before_action",
                float(info.get("projectile_total_count_before_action", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_hostile_count_before_action",
                float(info.get("projectile_hostile_count_before_action", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_owner_known_count",
                float(info.get("projectile_owner_known_count", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_present_before_action",
                float(bool(info.get("projectile_path_present_before_action"))),
            )
            self.logger.record_mean(
                "combat/projectile_path_count_before_action",
                float(info.get("projectile_path_count_before_action", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_requested_risk",
                float(info.get("projectile_path_requested_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_applied_risk",
                float(info.get("projectile_path_applied_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_risk_margin",
                float(info.get("projectile_path_risk_margin", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_path_action_improved",
                float(bool(info.get("projectile_path_action_improved"))),
            )
            for name, short in (
                ("projectile", "proj"),
                ("enemy", "enemy"),
                ("boundary", "bound"),
            ):
                self.logger.record_mean(
                    f"combat/{short}_path_min_risk",
                    float(info.get(f"{name}_path_min_risk", 0.0)),
                )
                self.logger.record_mean(
                    f"combat/{short}_path_unsafe_count",
                    float(info.get(f"{name}_path_unsafe_action_count", 0.0)),
                )
                self.logger.record_mean(
                    f"combat/{short}_path_unsafe_frac",
                    float(info.get(f"{name}_path_unsafe_action_fraction", 0.0)),
                )
                self.logger.record_mean(
                    f"combat/pre_{short}_path_min_risk",
                    float(info.get(f"{name}_path_min_risk_before_action", 0.0)),
                )
                self.logger.record_mean(
                    f"combat/pre_{short}_path_unsafe_frac",
                    float(info.get(f"{name}_path_unsafe_action_fraction_before_action", 0.0)),
                )
            self.logger.record_mean(
                "combat/hazard_override_rate",
                float(bool(info.get("hazard_overridden"))),
            )
            self.logger.record_mean(
                "combat/hazard_requested_risk",
                float(info.get("hazard_requested_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_stage_applied_risk",
                float(info.get("hazard_stage_applied_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_applied_risk",
                float(info.get("hazard_applied_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_risk_reduction",
                float(info.get("hazard_risk_reduction", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_enemy_risk",
                float(info.get("hazard_enemy_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_projectile_risk",
                float(info.get("hazard_projectile_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_indicator_risk",
                float(info.get("hazard_indicator_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_boundary_risk",
                float(info.get("hazard_boundary_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_applied_enemy_risk",
                float(info.get("hazard_applied_enemy_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_applied_projectile_risk",
                float(info.get("hazard_applied_projectile_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_applied_indicator_risk",
                float(info.get("hazard_applied_indicator_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_applied_boundary_risk",
                float(info.get("hazard_applied_boundary_risk", 0.0)),
            )
            hazard_source = str(info.get("hazard_source", "policy"))
            self.logger.record_mean(
                "combat/hazard_source_policy", float(hazard_source == "policy")
            )
            self.logger.record_mean(
                "combat/hazard_source_hazard", float(hazard_source == "hazard")
            )
            self.logger.record_mean(
                "combat/hazard_source_crowd_recovery",
                float(hazard_source == "crowd_recovery"),
            )
            self.logger.record_mean(
                "combat/hazard_recovery_active",
                float(bool(info.get("crowd_recovery_active"))),
            )
            self.logger.record_mean(
                "combat/hazard_state_interval_ms",
                float(info.get("hazard_state_interval_ms", 0.0)),
            )
            self.logger.record_mean(
                "combat/hazard_control_interval_ms",
                float(info.get("hazard_control_interval_ms", 0.0)),
            )
            for name in (
                "state_interval_p50_ms",
                "state_interval_p95_ms",
                "state_interval_p99_ms",
                "control_interval_p50_ms",
                "control_interval_p95_ms",
                "control_interval_p99_ms",
                "reward_time_scale",
            ):
                self.logger.record_mean(
                    f"control/{name}",
                    float(info.get(name, 0.0)),
                )
            for name in ("stale_state_count", "dropped_state_count", "state_tick_gap"):
                self.logger.record(
                    f"control/{name}",
                    float(info.get(name, 0.0)),
                )
            self.logger.record_mean(
                "combat/projectile_predicted_hazard_count",
                float(info.get("projectile_predicted_hazard_count", 0.0)),
            )
            self.logger.record_mean(
                "combat/projectile_nearest_tti",
                float(info.get("projectile_nearest_tti", -1.0)),
            )
            self.logger.record_mean(
                "combat/projectile_nearest_miss_distance",
                float(info.get("projectile_nearest_miss_distance", -1.0)),
            )
            visible = bool(info.get("projectile_visible_before_action"))
            tti = float(info.get("projectile_nearest_tti", -1.0))
            miss = float(info.get("projectile_nearest_miss_distance", -1.0))
            predicted = bool(info.get("projectile_hazard_exposed"))
            for key, condition, value in (
                ("visible", visible, float(info.get("damage_taken", 0.0))),
                ("tti", bool(0.0 <= tti <= 0.8), float(info.get("damage_taken", 0.0))),
                ("predicted_hazard", predicted, float(info.get("damage_taken", 0.0))),
                ("miss_distance", bool(miss >= 0.0), miss),
            ):
                if condition:
                    self.conditional_counts[key] = self.conditional_counts.get(key, 0) + 1
                    self.conditional_sums[key] = self.conditional_sums.get(key, 0.0) + value
            self.logger.record("combat/projectile_visible_exposure_count",
                               self.conditional_counts.get("visible", 0))
            self.logger.record("combat/projectile_tti_exposure_count",
                               self.conditional_counts.get("tti", 0))
            self.logger.record("combat/projectile_predicted_hazard_exposure_count",
                               self.conditional_counts.get("predicted_hazard", 0))
            self.logger.record_mean(
                "combat/projectile_tti_exposure_rate",
                float(bool(0.0 <= tti <= 0.8)),
            )
            self.logger.record_mean(
                "combat/projectile_predicted_hazard_exposure_rate",
                float(predicted),
            )
            self.logger.record_mean(
                "combat/projectile_tti_conditional_damage",
                self.conditional_sums.get("tti", 0.0)
                / max(1, self.conditional_counts.get("tti", 0)),
            )
            self.logger.record_mean(
                "combat/projectile_hazard_conditional_damage",
                self.conditional_sums.get("predicted_hazard", 0.0)
                / max(1, self.conditional_counts.get("predicted_hazard", 0)),
            )
            self.logger.record_mean(
                "combat/projectile_conditional_miss_distance",
                self.conditional_sums.get("miss_distance", 0.0)
                / max(1, self.conditional_counts.get("miss_distance", 0)),
            )
            self.logger.record_mean(
                "combat/damage_taken",
                float(info.get("damage_taken", 0.0)),
            )
            self.logger.record_mean(
                "combat/damage_after_projectile_visible",
                float(info.get("damage_after_projectile_visible", 0.0)),
            )
            self.logger.record_mean(
                "combat/damage_after_projectile_hazard",
                float(info.get("damage_after_projectile_hazard", 0.0)),
            )
            self.logger.record_mean(
                "combat/death_after_projectile_visible",
                float(bool(info.get("death_after_projectile_visible"))),
            )
            self.logger.record_mean(
                "combat/death_after_projectile_hazard",
                float(bool(info.get("death_after_projectile_hazard"))),
            )
            self.logger.record_mean(
                "combat/enemy_path_max_risk",
                float(info.get("enemy_path_max_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/enemy_path_action_risk",
                float(info.get("enemy_path_action_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/boundary_path_max_risk",
                float(info.get("boundary_path_max_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/boundary_path_action_risk",
                float(info.get("boundary_path_action_risk", 0.0)),
            )
            self.logger.record_mean(
                "combat/selected_path_risk_penalty",
                float(info.get("selected_path_risk_penalty", 0.0)),
            )
            self.logger.record_mean(
                "control/effective_state_hz",
                float(info.get("effective_state_hz", 0.0)),
            )
            self.logger.record_mean(
                "movement/distance",
                float(info.get("movement_distance", 0.0)),
            )
            self.logger.record_mean(
                "movement/efficiency",
                float(info.get("movement_efficiency", 0.0)),
            )
            self.logger.record_mean(
                "movement/reversal_rate",
                float(bool(info.get("movement_reversal"))),
            )
            self.logger.record_mean(
                "movement/low_motion_rate",
                float(bool(info.get("movement_low_motion"))),
            )
            self.logger.record_mean(
                "movement/center_stagnation_rate",
                float(bool(info.get("movement_center_stagnation"))),
            )
            self.logger.record_mean(
                "movement/idle_penalty",
                float(info.get("movement_idle_penalty", 0.0)),
            )
            self.logger.record_mean(
                "movement/reversal_penalty",
                float(info.get("movement_reversal_penalty", 0.0)),
            )
            self.logger.record_mean(
                "movement/low_motion_penalty",
                float(info.get("movement_low_motion_penalty", 0.0)),
            )
            self.logger.record_mean(
                "movement/center_stagnation_penalty",
                float(info.get("movement_center_stagnation_penalty", 0.0)),
            )
            self.logger.record_mean(
                "reward/total",
                float(info.get("reward_total", 0.0)),
            )
            reward_components = info.get("reward_components", {})
            if isinstance(reward_components, dict):
                for name, value in reward_components.items():
                    self.logger.record_mean(
                        f"reward/{name}",
                        float(value),
                    )
            for action in range(9):
                self.logger.record_mean(
                    f"actions/applied_{action}", float(applied == action)
                )
            if index < len(dones) and bool(dones[index]):
                self.logger.record_mean("combat/episode_wave", wave)
                reward_components = info.get("reward_components", {})
                death_reward = (
                    float(reward_components.get("death", 0.0))
                    if isinstance(reward_components, dict)
                    else 0.0
                )
                # A terminal episode with a death penalty is one actual player
                # death.  Count it once here, rather than once per dead-state
                # observation, and retain separate curves for each wave.
                if bool(info.get("dead")):
                    self.logger.record("combat/episode_death", 1.0)
                    self.logger.record("combat/episode_death_wave", max(0, wave))
                    self.deaths_by_wave[wave] = self.deaths_by_wave.get(wave, 0) + 1
                    self.total_deaths += 1
                    self.logger.record(
                        f"combat/deaths_wave_{max(0, wave)}",
                        self.deaths_by_wave[wave],
                    )
                    self.logger.record("combat/death_count_total", self.total_deaths)
                else:
                    self.logger.record("combat/episode_death", 0.0)
                victory = bool(info.get("victory"))
                wave_clear = bool(info.get("wave_clear"))
                self.logger.record("combat/episode_victory", float(victory))
                self.logger.record("combat/episode_wave_clear", float(wave_clear))
                self.total_victories += int(victory)
                self.total_wave_clears += int(wave_clear)
                self.logger.record("combat/victory_count_total", self.total_victories)
                self.logger.record("combat/wave_clear_count_total", self.total_wave_clears)
                self.episode_id += 1
        return True


def _anchor_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([record["features"] for record in records], dtype=np.float32)
    actions = np.asarray([int(record["action"]) for record in records], dtype=np.int64)
    return features, actions


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Fine-tune the human Brotato combat base with anchored PPO"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=cfg.output_dir / "human_combat_base_candidate.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=cfg.output_dir / "human_combat_v1.jsonl",
    )
    parser.add_argument("--timesteps", type=int, default=cfg.total_timesteps)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bc-coefficient", type=float, default=0.20)
    parser.add_argument("--bc-batches", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()

    cfg = replace(cfg, safety_shield=not args.no_safety)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    base, metadata = load_combat_base(args.base_model)
    records = load_records(args.dataset)
    if len(records) < 100:
        raise RuntimeError(f"only {len(records)} valid human records in {args.dataset}")
    features, actions = _anchor_arrays(records)

    env = Monitor(BrotatoApiEnv(cfg, vectorizer=RichCombatVectorizer()))
    checkpoints = cfg.output_dir / "human_finetune_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model = HumanAnchoredPPO.load(args.resume, env=env, device=args.device)
        transfer_difference = None
        print(f"[human-ppo] resumed={args.resume.resolve()}")
    else:
        model = HumanAnchoredPPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-5,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.002,
            tensorboard_log=str(cfg.output_dir / "logs"),
            device=args.device,
            bc_coefficient=args.bc_coefficient,
            bc_batches=args.bc_batches,
            policy_kwargs={
                "features_extractor_class": CombatLayerNormExtractor,
                "net_arch": {"pi": [128, 64], "vf": [128, 64]},
                "activation_fn": nn.Tanh,
            },
        )
        transfer_difference = initialize_actor_from_human_base(model, base)
    model.bc_coefficient = max(0.0, float(args.bc_coefficient))
    model.bc_batches = max(0, int(args.bc_batches))
    model.set_human_anchor(features, actions)
    print(
        f"[human-ppo] base={args.base_model.resolve()} records={len(records)} "
        f"validation_accuracy={metadata.get('validation_accuracy')} "
        f"transfer_max_abs_diff={transfer_difference} safety={cfg.safety_shield} "
        f"bc_coefficient={model.bc_coefficient}"
    )
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=20_000,
            save_path=str(checkpoints),
            name_prefix="human_base_ppo",
        ),
        SaveBestRollingRewardCallback(cfg.output_dir / "human_finetune_best", min_episodes=10),
        CombatTensorboardCallback(),
    ])
    try:
        model.learn(
            total_timesteps=max(1, int(args.timesteps)),
            callback=callbacks,
            tb_log_name="HumanBasePPO",
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        target = cfg.output_dir / "human_base_ppo_interrupted"
        model.save(str(target))
        print(f"[human-ppo] interrupted model saved={target}.zip")
        return 130
    except Exception:
        target = cfg.output_dir / "human_base_ppo_recovery"
        model.save(str(target))
        print(f"[human-ppo] error recovery model saved={target}.zip")
        raise
    finally:
        env.close()
    target = cfg.output_dir / "human_base_ppo_final"
    model.save(str(target))
    print(f"[human-ppo] final model saved={target}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
