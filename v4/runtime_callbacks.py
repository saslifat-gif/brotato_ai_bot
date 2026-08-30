"""Training callbacks that protect useful combat checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveBestRollingRewardCallback(BaseCallback):
    """Save whenever the rolling episode reward improves.

    This is a training proxy, not a replacement for deterministic evaluation.
    """

    def __init__(self, output_dir: Path, *, min_episodes: int = 10, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.output_dir = Path(output_dir)
        self.min_episodes = max(2, int(min_episodes))
        self.best_mean_reward = -float("inf")
        self.last_episode_count = -1

    def _on_step(self) -> bool:
        episodes = list(self.model.ep_info_buffer or [])
        episode_count = int(getattr(self.model, "_episode_num", len(episodes)))
        if len(episodes) < self.min_episodes or episode_count == self.last_episode_count:
            return True
        self.last_episode_count = episode_count
        window = episodes[-self.min_episodes:]
        mean_reward = float(np.mean([float(item["r"]) for item in window]))
        mean_length = float(np.mean([float(item["l"]) for item in window]))
        if mean_reward <= self.best_mean_reward:
            return True
        self.best_mean_reward = mean_reward
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "best_training_agent"
        self.model.save(str(target))
        metadata = {
            "timesteps": int(self.num_timesteps),
            "episodes_in_window": len(window),
            "mean_reward": mean_reward,
            "mean_episode_length": mean_length,
            "warning": "training proxy; validate with v4.run_frozen",
        }
        (self.output_dir / "best_training_agent.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if self.verbose:
            print(
                f"[v4-best] saved={target}.zip reward={mean_reward:.3f} "
                f"episode_length={mean_length:.1f} steps={self.num_timesteps}"
            )
        return True
