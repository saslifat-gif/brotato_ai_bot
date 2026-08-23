"""Train a recurrent movement policy from the structured Brotato API."""

import os

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from v3.config import load_config
from v3.env.brotato_api_env import BrotatoApiEnv


def main() -> int:
    if RecurrentPPO is None:
        raise RuntimeError("sb3-contrib is required: pip install sb3-contrib")
    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = cfg.output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    env = Monitor(BrotatoApiEnv(cfg))
    callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(checkpoints),
        name_prefix="api_recurrent_ppo",
    )
    resume = os.environ.get("BROTATO_V3_RESUME_MODEL", "").strip()
    if resume:
        model = RecurrentPPO.load(resume, env=env, device=os.environ.get("BROTATO_V3_DEVICE", "auto"))
        print(f"[v3-train] resumed={resume}")
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.02,
            tensorboard_log=str(cfg.output_dir / "logs"),
            device=os.environ.get("BROTATO_V3_DEVICE", "auto"),
            policy_kwargs={"lstm_hidden_size": 256, "n_lstm_layers": 1},
        )
    print(
        f"[v3-train] waiting on {cfg.host}:{cfg.port}; launch Brotato with the bridge mod enabled"
    )
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback,
            reset_num_timesteps=not bool(resume),
        )
    except KeyboardInterrupt:
        target = cfg.output_dir / "interrupted_agent"
        model.save(str(target))
        print(f"[v3-train] interrupted model saved={target}.zip")
        return 130
    except Exception:
        target = cfg.output_dir / "recovery_agent"
        model.save(str(target))
        print(f"[v3-train] error recovery model saved={target}.zip")
        raise
    finally:
        env.close()
    target = cfg.output_dir / "final_agent"
    model.save(str(target))
    print(f"[v3-train] final model saved={target}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
