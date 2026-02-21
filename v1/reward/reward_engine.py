from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RewardBreakdown:
    alive_reward: float
    time_penalty: float        # per-step survival cost (hunger games)
    non_battle_penalty: float
    damage_penalty: float
    idle_penalty: float
    activity_bonus: float
    loot_collect_bonus: float
    kill_bonus: float          # loot-spawn proxy for enemy kill
    death_penalty: float
    total: float


class RewardEngine:
    def __init__(
        self,
        alive_reward: float,
        non_battle_penalty: float,
        damage_scale: float,
        idle_penalty: float,
        activity_bonus: float,
        loot_bonus: float,
        death_penalty: float,
        idle_diff_threshold: float,
        loot_delta_trigger: float,
        shop_penalty_grace_sec: float = 30.0,
        time_penalty: float = 0.0,
        kill_bonus: float = 0.0,
        kill_spawn_trigger: float = 0.015,
    ):
        self.alive_reward = float(alive_reward)
        # Per-step survival cost: positive value → subtracted each battle step.
        # "Hunger Games" mode: being alive costs points, forcing the AI to earn
        # positive rewards (loot / activity) or die trying.
        self.time_penalty = float(max(0.0, time_penalty))
        self.non_battle_penalty = float(non_battle_penalty)
        self.damage_scale = float(damage_scale)
        self.idle_penalty = float(idle_penalty)
        self.activity_bonus = float(activity_bonus)
        self.loot_bonus = float(loot_bonus)
        self.death_penalty = float(death_penalty)
        self.idle_diff_threshold = float(idle_diff_threshold)
        self.loot_delta_trigger = float(max(1e-6, loot_delta_trigger))
        self.shop_penalty_grace_sec = float(max(0.0, shop_penalty_grace_sec))
        # Kill reward: triggered when loot_spawn (yellow pixels appearing near
        # player) exceeds kill_spawn_trigger — proxy for an enemy dying nearby.
        self.kill_bonus = float(max(0.0, kill_bonus))
        self.kill_spawn_trigger = float(max(1e-6, kill_spawn_trigger))

        self.reward_components_last: Dict[str, float] = {}
        self.episode_component_sums: Dict[str, float] = {}
        self.loot_events = 0
        self.kill_events = 0
        self.death_events = 0
        self.reset_episode()

    def reset_episode(self):
        self.reward_components_last = {
            "alive_reward": 0.0,
            "time_penalty": 0.0,
            "non_battle_penalty": 0.0,
            "damage_penalty": 0.0,
            "idle_penalty": 0.0,
            "activity_bonus": 0.0,
            "loot_collect_bonus": 0.0,
            "kill_bonus": 0.0,
            "death_penalty": 0.0,
            "total": 0.0,
        }
        self.episode_component_sums = {k: 0.0 for k in self.reward_components_last.keys()}
        self.loot_events = 0
        self.kill_events = 0
        self.death_events = 0

    def compute(
        self,
        prev_hp: float,
        curr_hp: float,
        is_battle: bool,
        obs_diff: float,
        loot_delta: float,
        dead: bool,
        state_name: str = "",
        state_elapsed_sec: float = 0.0,
        is_moving: bool = True,
        loot_spawn: float = 0.0,
    ) -> RewardBreakdown:
        hp_drop = max(0.0, float(prev_hp - curr_hp))
        damage_penalty = self.damage_scale * hp_drop

        # alive_reward is now intentionally tiny (or 0) — the real driver is
        # time_penalty which makes every idle battle step cost the AI points.
        alive_reward = self.alive_reward if is_battle else 0.0
        # Per-step survival cost: subtracted every battle step.
        # Forces the AI out of "survival mode" — it must collect loot / stay
        # active to offset this constant drain.
        time_step_penalty = self.time_penalty if is_battle else 0.0
        non_battle_penalty = self.non_battle_penalty if not is_battle else 0.0
        if (not is_battle) and str(state_name or "").strip().lower() == "shop":
            if float(state_elapsed_sec) < float(self.shop_penalty_grace_sec):
                non_battle_penalty = 0.0

        idle_penalty = 0.0
        activity_bonus = 0.0
        if is_battle:
            # Use actual movement action as primary idle signal — obs_diff alone is
            # unreliable because enemy movement keeps obs_diff high even when the
            # player is standing still (e.g. hugging a corner).
            if not is_moving:
                # Player chose not to press any movement key
                idle_penalty = self.idle_penalty
            elif obs_diff < self.idle_diff_threshold:
                # Player is pressing a key but the scene isn't changing much
                # (e.g. stuck against a wall) — apply a reduced penalty
                idle_penalty = self.idle_penalty * 0.5
            else:
                scale = min(1.0, (obs_diff - self.idle_diff_threshold) / max(1.0, self.idle_diff_threshold))
                activity_bonus = self.activity_bonus * float(scale)

        loot_collect_bonus = 0.0
        if is_battle and loot_delta > self.loot_delta_trigger:
            scale = min(1.0, float(loot_delta) / float(self.loot_delta_trigger))
            loot_collect_bonus = self.loot_bonus * float(scale)
            if loot_collect_bonus > 0.0:
                self.loot_events += 1

        # Kill bonus: loot_spawn > threshold means yellow XP/loot pixels
        # suddenly appeared near the player → enemy killed nearby.
        kill_bonus_val = 0.0
        if is_battle and self.kill_bonus > 0.0 and float(loot_spawn) > self.kill_spawn_trigger:
            kill_bonus_val = self.kill_bonus
            self.kill_events += 1

        death_penalty = self.death_penalty if dead else 0.0
        if death_penalty > 0.0:
            self.death_events += 1

        total = 0.0
        total += alive_reward
        total -= time_step_penalty   # hunger-games drain
        total -= non_battle_penalty
        total -= damage_penalty
        total -= idle_penalty
        total += activity_bonus
        total += loot_collect_bonus
        total += kill_bonus_val
        total -= death_penalty

        def _safe(v: float) -> float:
            return float(v) if np.isfinite(v) else 0.0

        alive_reward = _safe(alive_reward)
        time_step_penalty = _safe(time_step_penalty)
        non_battle_penalty = _safe(non_battle_penalty)
        damage_penalty = _safe(damage_penalty)
        idle_penalty = _safe(idle_penalty)
        activity_bonus = _safe(activity_bonus)
        loot_collect_bonus = _safe(loot_collect_bonus)
        kill_bonus_val = _safe(kill_bonus_val)
        death_penalty = _safe(death_penalty)
        total = _safe(total)

        out = RewardBreakdown(
            alive_reward=alive_reward,
            time_penalty=time_step_penalty,
            non_battle_penalty=non_battle_penalty,
            damage_penalty=damage_penalty,
            idle_penalty=idle_penalty,
            activity_bonus=activity_bonus,
            loot_collect_bonus=loot_collect_bonus,
            kill_bonus=kill_bonus_val,
            death_penalty=death_penalty,
            total=total,
        )

        self.reward_components_last = {
            "alive_reward": out.alive_reward,
            "time_penalty": out.time_penalty,
            "non_battle_penalty": out.non_battle_penalty,
            "damage_penalty": out.damage_penalty,
            "idle_penalty": out.idle_penalty,
            "activity_bonus": out.activity_bonus,
            "loot_collect_bonus": out.loot_collect_bonus,
            "kill_bonus": out.kill_bonus,
            "death_penalty": out.death_penalty,
            "total": out.total,
        }
        for k, v in self.reward_components_last.items():
            self.episode_component_sums[k] = float(self.episode_component_sums.get(k, 0.0) + float(v))

        return out
