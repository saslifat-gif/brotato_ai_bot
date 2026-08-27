"""One validated configuration object for the active runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from brotato_ai.bridge.protocol import DEFAULT_HOST, DEFAULT_PORT


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
    control_hz: float = 24.0
    recorder_hz: float = 60.0
    cache_max_gib: float = 10.0

    def validate(self) -> "RuntimeConfig":
        if not self.host.strip():
            raise ValueError("bridge host cannot be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError(f"bridge port out of range: {self.port}")
        if not 4.0 <= float(self.control_hz) <= 24.0:
            raise ValueError(f"control_hz must be between 4 and 24: {self.control_hz}")
        if not 1.0 <= float(self.recorder_hz) <= 120.0:
            raise ValueError(f"recorder_hz must be between 1 and 120: {self.recorder_hz}")
        if not 0.1 <= float(self.cache_max_gib) <= 10.0:
            raise ValueError(f"cache_max_gib must be between 0.1 and 10: {self.cache_max_gib}")
        if self.state_timeout_sec < 1.0 or self.reset_timeout_sec < 10.0:
            raise ValueError("bridge timeouts are below safe minimums")
        return self

    def startup_summary(self) -> str:
        model = str(self.ui_model_path) if self.ui_model_path else "auto/none"
        return (
            "[config] "
            f"bridge={self.host}:{self.port} control_hz={self.control_hz:g} "
            f"recorder_hz={self.recorder_hz:g} safety={self.safety_shield} "
            f"late_wave={self.late_wave_focus} menus={self.automate_menus} "
            f"profile={self.ui_build_profile} ui_model={model} "
            f"output={self.output_dir} cache_max_gib={self.cache_max_gib:g}"
        )


V3Config = RuntimeConfig


def load_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = os.environ if environ is None else environ
    root = Path(__file__).resolve().parents[3]
    output = Path(env.get("BROTATO_V3_OUTPUT_DIR", str(root / "models" / "version_3")))
    if not output.is_absolute():
        output = (root / output).resolve()
    ui_build_profile = env.get("BROTATO_V3_UI_BUILD_PROFILE", "ranged_smg").strip().lower()
    ui_model_value = env.get("BROTATO_V3_UI_MODEL", "").strip()
    if ui_model_value:
        ui_model_path = Path(ui_model_value).resolve()
    else:
        candidate_names = [f"ui_build_base_{ui_build_profile}_candidate.pt"]
        if ui_build_profile == "stick_melee":
            candidate_names.append("ui_build_base_v3_candidate.pt")
        candidate = next((output / name for name in candidate_names if (output / name).exists()), None)
        ui_model_path = candidate.resolve() if candidate is not None else None
    ui_log_value = env.get(
        "BROTATO_V3_UI_DATASET", str(output / f"ui_decisions_{ui_build_profile}_v1.jsonl")
    ).strip()
    ui_decision_log = (
        None
        if ui_log_value.lower() in {"", "0", "off", "none"}
        else Path(ui_log_value).resolve()
    )
    combat_log_value = env.get("BROTATO_V3_COMBAT_DATASET", "").strip()
    combat_decision_log = (
        None
        if combat_log_value.lower() in {"", "0", "off", "none"}
        else Path(combat_log_value).resolve()
    )
    return RuntimeConfig(
        host=env.get("BROTATO_V3_HOST", DEFAULT_HOST),
        port=int(env.get("BROTATO_V3_PORT", str(DEFAULT_PORT))),
        state_timeout_sec=max(1.0, float(env.get("BROTATO_V3_STATE_TIMEOUT", "30"))),
        reset_timeout_sec=max(10.0, float(env.get("BROTATO_V3_RESET_TIMEOUT", "600"))),
        output_dir=output,
        total_timesteps=max(1, int(env.get("BROTATO_V3_TIMESTEPS", "1000000"))),
        automate_menus=_enabled(env.get("BROTATO_V3_AUTOMATE_MENUS", "1"), default=True),
        max_shop_buys=max(0, int(env.get("BROTATO_V3_MAX_SHOP_BUYS", "4"))),
        max_shop_rerolls=max(0, int(env.get("BROTATO_V3_MAX_SHOP_REROLLS", "1"))),
        ui_build_profile=ui_build_profile,
        ui_model_path=ui_model_path,
        ui_decision_log=ui_decision_log,
        safety_shield=_enabled(env.get("BROTATO_V3_SAFETY_SHIELD", "0"), default=False),
        combat_decision_log=combat_decision_log,
        late_wave_focus=_enabled(env.get("BROTATO_V3_LATE_WAVE_FOCUS", "0"), default=False),
        control_hz=float(env.get("BROTATO_V4_CONTROL_HZ", "24")),
        recorder_hz=float(env.get("BROTATO_V4_RECORDER_HZ", "60")),
        cache_max_gib=float(env.get("BROTATO_V4_CACHE_MAX_GIB", "10")),
    ).validate()
