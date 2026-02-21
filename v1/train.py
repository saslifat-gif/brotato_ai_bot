import os
from pathlib import Path
from typing import Callable

import cv2
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from config.runtime_config import load_runtime_config
from env.brotato_env import BrotatoEnv
from runtime.stop_manager import (
    StopManager,
    StopTrainingCallback,
    TrainingStopRequested,
    install_stop_handlers,
)


def linear_schedule(lr_start: float, lr_end: float) -> Callable[[float], float]:
    """Return a SB3-compatible LR schedule: progress 1.0→0.0 maps lr_start→lr_end."""
    def schedule(progress_remaining: float) -> float:
        return lr_end + progress_remaining * (lr_start - lr_end)
    return schedule


def find_latest_checkpoint(models_dir: Path, prefix: str = "brotato_model_") -> str:
    if not models_dir.exists():
        return ""
    best_path = ""
    best_step = -1
    for name in os.listdir(models_dir):
        if not (name.startswith(prefix) and name.endswith("_steps.zip")):
            continue
        try:
            step = int(name[len(prefix) : -len("_steps.zip")])
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best_path = str(models_dir / name)
    return best_path


def resolve_resume_path(resume_cfg: str, script_dir: Path, models_dir: Path) -> str:
    value = str(resume_cfg or "").strip()
    if not value:
        return ""
    if value.lower() == "auto":
        return find_latest_checkpoint(models_dir=models_dir)
    p = Path(value)
    if not p.is_absolute():
        p = (script_dir / p).resolve()
    return str(p)


def build_or_resume_model(cfg, env, resume_path: str, logs_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr_schedule = linear_schedule(cfg.ppo_learning_rate, cfg.ppo_lr_end)
    model = None

    if resume_path and os.path.exists(resume_path):
        print(f"[resume] loading model from: {resume_path}")
        try:
            model = PPO.load(resume_path, env=env, device=device)
            # Check if hyperparameters differ from config
            hp_changed = (
                model.n_steps != cfg.ppo_n_steps
                or model.batch_size != cfg.ppo_batch_size
                or model.n_epochs != cfg.ppo_n_epochs
                or model.ent_coef != cfg.ppo_ent_coef
            )
            if hp_changed:
                print(
                    f"[resume] hyperparams changed, rebuilding model with new config "
                    f"(n_steps: {model.n_steps}->{cfg.ppo_n_steps}, "
                    f"batch_size: {model.batch_size}->{cfg.ppo_batch_size}, "
                    f"lr: linear {cfg.ppo_learning_rate}->{cfg.ppo_lr_end})"
                )
                old_policy_state = model.policy.state_dict()
                model = PPO(
                    "CnnPolicy",
                    env,
                    verbose=1,
                    learning_rate=lr_schedule,
                    ent_coef=cfg.ppo_ent_coef,
                    n_steps=cfg.ppo_n_steps,
                    batch_size=cfg.ppo_batch_size,
                    n_epochs=cfg.ppo_n_epochs,
                    tensorboard_log=str(logs_dir),
                    device=device,
                )
                model.policy.load_state_dict(old_policy_state)
                print("[resume] policy weights transferred to new model")
            else:
                # Even on clean resume, switch to linear schedule going forward.
                model.lr_schedule = lr_schedule
        except Exception as e:
            msg = str(e)
            if "Observation spaces do not match" in msg or "Action spaces do not match" in msg:
                print(f"[resume] incompatible checkpoint: {e}")
                print("[resume] start from scratch")
                model = None
            else:
                raise
    elif resume_path:
        print(f"[resume] model not found: {resume_path}, start from scratch")

    if model is None:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=lr_schedule,
            ent_coef=cfg.ppo_ent_coef,
            n_steps=cfg.ppo_n_steps,
            batch_size=cfg.ppo_batch_size,
            n_epochs=cfg.ppo_n_epochs,
            tensorboard_log=str(logs_dir),
            device=device,
        )
    return model


def main():
    cfg = load_runtime_config()
    stop_manager = StopManager()
    install_stop_handlers(stop_manager)

    script_dir = Path(__file__).resolve().parent
    output_root = str(os.environ.get("BROTATO_OUTPUT_DIR", script_dir.parent / "models" / "version_1")).strip()
    models_dir = Path(output_root).resolve()
    logs_dir = models_dir / "ppo_brotato_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[path] output models_dir={models_dir}")
    print(f"[path] tensorboard logs_dir={logs_dir}")

    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    if cfg.torch_interop_threads > 0:
        try:
            torch.set_num_interop_threads(cfg.torch_interop_threads)
        except Exception:
            pass

    env = BrotatoEnv(cfg=cfg, stop_manager=stop_manager)

    resume_path = resolve_resume_path(cfg.resume_model, script_dir, models_dir)
    model = build_or_resume_model(cfg, env, resume_path, logs_dir)

    checkpoint_cb = CheckpointCallback(
        save_freq=5000,
        save_path=str(models_dir),
        name_prefix="brotato_model",
    )
    stop_cb = StopTrainingCallback(stop_manager=stop_manager)
    callbacks = CallbackList([checkpoint_cb, stop_cb])

    print(
        "AI training started. "
        f"PPO cfg: n_steps={cfg.ppo_n_steps}, batch_size={cfg.ppo_batch_size}, "
        f"n_epochs={cfg.ppo_n_epochs}, lr={cfg.ppo_learning_rate}->{cfg.ppo_lr_end}(linear), "
        f"ent_coef={cfg.ppo_ent_coef}, "
        f"total_timesteps={cfg.ppo_total_timesteps}, reset_num_timesteps={cfg.reset_num_timesteps}, "
        f"progress_bar={cfg.ppo_progress_bar}"
    )
    print(
        "[reward] "
        f"alive={cfg.reward_alive} non_battle={cfg.reward_non_battle_penalty} "
        f"damage_scale={cfg.reward_damage_scale} idle={cfg.reward_idle_penalty} "
        f"activity={cfg.reward_activity_bonus} loot={cfg.reward_loot_bonus} death={cfg.reward_death_penalty}"
    )

    interrupted = False
    interrupt_reason = ""
    try:
        try:
            model.learn(
                total_timesteps=cfg.ppo_total_timesteps,
                callback=callbacks,
                reset_num_timesteps=cfg.reset_num_timesteps,
                progress_bar=cfg.ppo_progress_bar,
            )
        except Exception as e:
            msg = str(e).lower()
            if cfg.ppo_progress_bar and ("tqdm" in msg or "rich" in msg or "progress" in msg):
                print(f"[train] progress bar unavailable, fallback without progress bar: {e}")
                model.learn(
                    total_timesteps=cfg.ppo_total_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=cfg.reset_num_timesteps,
                )
            else:
                raise
        if stop_manager.should_stop():
            interrupted = True
            interrupt_reason = stop_manager.reason() or "stop_requested"
            raise TrainingStopRequested(interrupt_reason)
    except TrainingStopRequested as e:
        interrupted = True
        interrupt_reason = str(e)
        print(f"[control] training stop requested: {interrupt_reason}")
    except KeyboardInterrupt:
        interrupted = True
        interrupt_reason = "KeyboardInterrupt"
        print("[control] KeyboardInterrupt")
    finally:
        try:
            if interrupted:
                interrupted_path = models_dir / "brotato_interrupted_agent"
                model.save(str(interrupted_path))
                print(f"Training stopped. Progress saved: {interrupted_path}.zip reason={interrupt_reason}")
            else:
                final_path = models_dir / "brotato_final_agent"
                model.save(str(final_path))
                print(f"Training completed. Model saved: {final_path}.zip")
        except Exception as e:
            print(f"[save] failed: {e}")
        try:
            env.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
