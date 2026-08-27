"""Runtime callbacks owned by the active training package."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class GracefulStopCallback(BaseCallback):
    """End ``learn`` cleanly when the operator creates a stop-request file."""

    def __init__(self, request_path: Path, *, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.request_path = Path(request_path)

    def _on_training_start(self) -> None:
        try:
            self.request_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _on_step(self) -> bool:
        if not self.request_path.exists():
            return True
        if self.verbose:
            print(
                f"[training] graceful stop requested={self.request_path}; "
                "finishing rollout and saving final checkpoint",
                flush=True,
            )
        return False

