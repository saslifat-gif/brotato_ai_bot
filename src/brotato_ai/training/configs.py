"""One validated configuration object for the active runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from brotato_ai.bridge.protocol import DEFAULT_HOST, DEFAULT_PORT
from brotato_ai.policy.modes import DEFAULT_POLICY_MODE, PolicyMode, parse_policy_mode
from brotato_ai.ui.modes import (
    DEFAULT_BUILD_POLICY_MODE,
    BuildPolicyMode,
    parse_build_policy_mode,
)


def _enabled(value: str, *, default: bool) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RuntimeConfig:
    host: str
    port: int
    state_timeout_sec: float
    reset_timeout_sec: float
    output_dir: Path
    total_timesteps: int
    automate_menus: bool
    max_shop_buys: int
    max_shop_rerolls: int
    ui_build_profile: str
    ui_model_path: Path | None
    ui_decision_log: Path | None
    safety_shield: bool
    combat_decision_log: Path | None
    late_wave_focus: bool
    ui_model_explicit: bool = False
    build_policy_mode: BuildPolicyMode = DEFAULT_BUILD_POLICY_MODE
    control_hz: float = 24.0
    recorder_hz: float = 60.0
    cache_max_gib: float = 10.0
    runtime_profile_path: Path | None = None
    runtime_profile_sample_limit: int = 20_000
    # Learned human-policy integration.  Defaults keep production on the
    # unchanged handcrafted path.
    policy_mode: PolicyMode = DEFAULT_POLICY_MODE
    human_model_path: Path | None = None
    human_confidence_threshold: float = 0.35
    human_hold_prior_ms: float = 438.0
    human_decision_interval_ms: float = 438.0
    human_feature_schema_version: int = 1
    allow_experimental_full_learned: bool = False

    def validate(self) -> "RuntimeConfig":
        if not self.host.strip():
            raise ValueError("bridge host cannot be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError(f"bridge port out of range: {self.port}")
        if not 4.0 <= float(self.control_hz) <= 60.0:
            raise ValueError(f"control_hz must be between 4 and 60: {self.control_hz}")
        if not 1.0 <= float(self.recorder_hz) <= 120.0:
            raise ValueError(f"recorder_hz must be between 1 and 120: {self.recorder_hz}")
        if not 0.1 <= float(self.cache_max_gib) <= 10.0:
            raise ValueError(f"cache_max_gib must be between 0.1 and 10: {self.cache_max_gib}")
        if int(self.runtime_profile_sample_limit) < 100:
            raise ValueError("runtime_profile_sample_limit must be at least 100")
        if self.state_timeout_sec < 1.0 or self.reset_timeout_sec < 10.0:
            raise ValueError("bridge timeouts are below safe minimums")
        if not 0.0 <= float(self.human_confidence_threshold) <= 1.0:
            raise ValueError(
                f"human_confidence_threshold must be within [0, 1]: {self.human_confidence_threshold}"
            )
        if float(self.human_hold_prior_ms) < 50.0 or float(self.human_decision_interval_ms) < 50.0:
            raise ValueError("human hold/decision intervals must be at least 50 ms")
        if self.policy_mode in {PolicyMode.SHADOW_HUMAN, PolicyMode.HYBRID_HUMAN}:
            if self.human_model_path is None:
                raise ValueError(
                    f"{self.policy_mode.value} requires a human model path "
                    "(BROTATO_V4_HUMAN_MODEL)"
                )
        if self.policy_mode is PolicyMode.EXPERIMENTAL_FULL_LEARNED:
            if not self.allow_experimental_full_learned:
                raise ValueError(
                    "EXPERIMENTAL_FULL_LEARNED requires explicit opt-in "
                    "(BROTATO_V4_ALLOW_FULL_LEARNED=1)"
                )
            if self.human_model_path is None:
                raise ValueError("EXPERIMENTAL_FULL_LEARNED requires a human model path")
        if self.build_policy_mode is BuildPolicyMode.LEARNED:
            if self.ui_model_path is None or not self.ui_model_explicit:
                raise ValueError(
                    "LEARNED build policy requires an explicitly configured model "
                    "(BROTATO_V4_UI_MODEL); auto-discovered candidate checkpoints are "
                    "refused so an undertrained model cannot be deployed silently"
                )
        return self

    def startup_summary(self) -> str:
        model = str(self.ui_model_path) if self.ui_model_path else "auto/none"
        human_model = str(self.human_model_path) if self.human_model_path else "none"
        return (
            "[config] "
            f"bridge={self.host}:{self.port} control_hz={self.control_hz:g} "
            f"recorder_hz={self.recorder_hz:g} safety={self.safety_shield} "
            f"late_wave={self.late_wave_focus} menus={self.automate_menus} "
            f"profile={self.ui_build_profile} ui_model={model} "
            f"build_policy={self.build_policy_mode.value} "
            f"policy_mode={self.policy_mode.value} human_model={human_model} "
            f"human_confidence={self.human_confidence_threshold:g} "
            f"output={self.output_dir} cache_max_gib={self.cache_max_gib:g} "
            f"runtime_profile={self.runtime_profile_path or 'off'}"
        )


def load_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = os.environ if environ is None else environ
    root = Path(__file__).resolve().parents[3]
    output = Path(env.get("BROTATO_V4_OUTPUT_DIR", str(root / "models" / "version_3")))
    if not output.is_absolute():
        output = (root / output).resolve()
    ui_build_profile = env.get("BROTATO_V4_UI_BUILD_PROFILE", "ranged_smg").strip().lower()
    ui_model_value = env.get("BROTATO_V4_UI_MODEL", "").strip()
    ui_model_explicit = bool(ui_model_value)
    if ui_model_value:
        ui_model_path = Path(ui_model_value).resolve()
    else:
        candidate_names = [f"ui_build_base_{ui_build_profile}_candidate.pt"]
        if ui_build_profile == "stick_melee":
            candidate_names.append("ui_build_base_v3_candidate.pt")
        candidate = next((output / name for name in candidate_names if (output / name).exists()), None)
        ui_model_path = candidate.resolve() if candidate is not None else None
    ui_log_value = env.get(
        "BROTATO_V4_UI_DATASET", str(output / f"ui_decisions_{ui_build_profile}_v1.jsonl")
    ).strip()
    ui_decision_log = (
        None
        if ui_log_value.lower() in {"", "0", "off", "none"}
        else Path(ui_log_value).resolve()
    )
    combat_log_value = env.get("BROTATO_V4_COMBAT_DATASET", "").strip()
    combat_decision_log = (
        None
        if combat_log_value.lower() in {"", "0", "off", "none"}
        else Path(combat_log_value).resolve()
    )
    runtime_profile_value = env.get("BROTATO_RUNTIME_PROFILE_PATH", "").strip()
    runtime_profile_path = (
        None
        if runtime_profile_value.lower() in {"", "0", "off", "none"}
        else Path(runtime_profile_value).resolve()
    )
    return RuntimeConfig(
        host=env.get("BROTATO_V4_HOST", DEFAULT_HOST),
        port=int(env.get("BROTATO_V4_PORT", str(DEFAULT_PORT))),
        state_timeout_sec=max(1.0, float(env.get("BROTATO_V4_STATE_TIMEOUT", "30"))),
        reset_timeout_sec=max(10.0, float(env.get("BROTATO_V4_RESET_TIMEOUT", "600"))),
        output_dir=output,
        total_timesteps=max(1, int(env.get("BROTATO_V4_TIMESTEPS", "1000000"))),
        automate_menus=_enabled(env.get("BROTATO_V4_AUTOMATE_MENUS", "1"), default=True),
        max_shop_buys=max(0, int(env.get("BROTATO_V4_MAX_SHOP_BUYS", "4"))),
        max_shop_rerolls=max(0, int(env.get("BROTATO_V4_MAX_SHOP_REROLLS", "1"))),
        ui_build_profile=ui_build_profile,
        ui_model_path=ui_model_path,
        ui_model_explicit=ui_model_explicit,
        build_policy_mode=parse_build_policy_mode(
            env.get("BROTATO_V4_BUILD_POLICY_MODE", "")
        ),
        ui_decision_log=ui_decision_log,
        safety_shield=_enabled(env.get("BROTATO_V4_SAFETY_SHIELD", "0"), default=False),
        combat_decision_log=combat_decision_log,
        late_wave_focus=_enabled(env.get("BROTATO_V4_LATE_WAVE_FOCUS", "0"), default=False),
        control_hz=float(env.get("BROTATO_V4_CONTROL_HZ", "24")),
        recorder_hz=float(env.get("BROTATO_V4_RECORDER_HZ", "60")),
        cache_max_gib=float(env.get("BROTATO_V4_CACHE_MAX_GIB", "10")),
        runtime_profile_path=runtime_profile_path,
        runtime_profile_sample_limit=max(
            100, int(env.get("BROTATO_RUNTIME_PROFILE_SAMPLES", "20000"))
        ),
        policy_mode=parse_policy_mode(env.get("BROTATO_V4_POLICY_MODE", "")),
        human_model_path=(
            Path(env.get("BROTATO_V4_HUMAN_MODEL", "")).resolve()
            if env.get("BROTATO_V4_HUMAN_MODEL", "").strip()
            else None
        ),
        human_confidence_threshold=float(env.get("BROTATO_V4_HUMAN_CONFIDENCE", "0.35")),
        human_hold_prior_ms=float(env.get("BROTATO_V4_HUMAN_HOLD_MS", "438")),
        human_decision_interval_ms=float(env.get("BROTATO_V4_HUMAN_INTERVAL_MS", "438")),
        human_feature_schema_version=int(env.get("BROTATO_V4_HUMAN_FEATURE_SCHEMA", "1")),
        allow_experimental_full_learned=_enabled(
            env.get("BROTATO_V4_ALLOW_FULL_LEARNED", "0"), default=False
        ),
    ).validate()
