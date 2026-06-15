from collections import deque
from typing import Deque

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeKpiCallback(BaseCallback):
    """Logs game-outcome KPIs (survival time, waves, kills) to TensorBoard.

    The environment attaches an ``episode_kpi`` dict to ``info`` on the final
    step of each episode. These KPIs measure how well the agent actually plays
    Brotato, independent of the engineered reward — so reward-shaping changes
    can be evaluated against ground truth rather than against the reward signal
    they directly inflate.
    """

    KPI_KEYS = (
        "survival_time_sec",
        "survival_steps",
        "episode_reward",
        "waves_completed",
        "kills",
        "loot_events",
    )

    def __init__(self, window: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.window = int(max(1, window))
        self.episode_count = 0
        self._recent: dict[str, Deque[float]] = {k: deque(maxlen=self.window) for k in self.KPI_KEYS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            kpi = info.get("episode_kpi") if isinstance(info, dict) else None
            if not kpi:
                continue
            self.episode_count += 1
            for key in self.KPI_KEYS:
                val = float(kpi.get(key, 0.0))
                self._recent[key].append(val)
                # Per-episode value and a rolling mean for a smoother curve.
                self.logger.record(f"kpi/{key}", val)
                window_vals = self._recent[key]
                self.logger.record(f"kpi_mean/{key}", sum(window_vals) / len(window_vals))
            self.logger.record("kpi/episodes", self.episode_count)
            if self.verbose > 0:
                print(
                    f"[kpi] ep={self.episode_count} "
                    f"waves={int(kpi.get('waves_completed', 0))} "
                    f"survival={float(kpi.get('survival_time_sec', 0.0)):.1f}s "
                    f"steps={int(kpi.get('survival_steps', 0))} "
                    f"kills={int(kpi.get('kills', 0))} "
                    f"loot={int(kpi.get('loot_events', 0))} "
                    f"reward={float(kpi.get('episode_reward', 0.0)):.1f}"
                )
        return True
