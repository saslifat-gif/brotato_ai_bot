"""Train the detector-driven v2 policy with RecurrentPPO."""

import os
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from v2.config import load_config
from v2.env import BrotatoVectorEnv


def main() -> int:
    if RecurrentPPO is None:
        raise RuntimeError("sb3-contrib is required: pip install sb3-contrib")
    cfg = load_config()
    if not cfg.combat_weights.exists():
        raise RuntimeError(
            f"combat weights are missing: {cfg.combat_weights}\n"
            "Run record_v2.bat, label the dataset, and train the combat detector first."
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(BrotatoVectorEnv(cfg))
    checkpoint = CheckpointCallback(
        save_freq=5000,
        save_path=str(cfg.output_dir / "checkpoints"),
        name_prefix="recurrent_ppo",
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        ent_coef=0.02,
        tensorboard_log=str(cfg.output_dir / "logs"),
        device=os.environ.get("BROTATO_V2_POLICY_DEVICE", "auto"),
        policy_kwargs={"lstm_hidden_size": 256, "n_lstm_layers": 1},
    )
    try:
        model.learn(
            total_timesteps=int(os.environ.get("BROTATO_V2_TOTAL_TIMESTEPS", "1000000")),
            callback=checkpoint,
        )
    except KeyboardInterrupt:
        model.save(str(cfg.output_dir / "interrupted_agent"))
        print(f"[v2] interrupted model saved to {cfg.output_dir / 'interrupted_agent.zip'}")
        return 130
    finally:
        env.close()
    model.save(str(cfg.output_dir / "final_agent"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

