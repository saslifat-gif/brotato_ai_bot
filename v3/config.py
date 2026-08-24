"""Configuration for the API-first trainer."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from v3.protocol import DEFAULT_HOST, DEFAULT_PORT


@dataclass(frozen=True)
class V3Config:
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
    ui_model_path: Optional[Path]
    ui_decision_log: Optional[Path]
    safety_shield: bool
    combat_decision_log: Optional[Path]


def load_config() -> V3Config:
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("BROTATO_V3_OUTPUT_DIR", root / "models" / "version_3"))
    if not output.is_absolute():
        output = (root / output).resolve()
    ui_model_value = os.environ.get("BROTATO_V3_UI_MODEL", "").strip()
    if ui_model_value:
        ui_model_path = Path(ui_model_value).resolve()
    else:
        trained_ui_candidate = output / "ui_build_base_v3_candidate.pt"
        ui_model_path = (
            trained_ui_candidate.resolve() if trained_ui_candidate.exists() else None
        )
    ui_log_value = os.environ.get(
        "BROTATO_V3_UI_DATASET", str(output / "ui_decisions_stick_melee_v3.jsonl")
    ).strip()
    ui_decision_log = (
        None
        if ui_log_value.lower() in {"", "0", "off", "none"}
        else Path(ui_log_value).resolve()
    )
    combat_log_value = os.environ.get("BROTATO_V3_COMBAT_DATASET", "").strip()
    combat_decision_log = (
        None
        if combat_log_value.lower() in {"", "0", "off", "none"}
        else Path(combat_log_value).resolve()
    )
    return V3Config(
        host=os.environ.get("BROTATO_V3_HOST", DEFAULT_HOST),
        port=int(os.environ.get("BROTATO_V3_PORT", str(DEFAULT_PORT))),
        state_timeout_sec=max(1.0, float(os.environ.get("BROTATO_V3_STATE_TIMEOUT", "30"))),
        reset_timeout_sec=max(10.0, float(os.environ.get("BROTATO_V3_RESET_TIMEOUT", "600"))),
        output_dir=output,
        total_timesteps=max(1, int(os.environ.get("BROTATO_V3_TIMESTEPS", "1000000"))),
        automate_menus=os.environ.get("BROTATO_V3_AUTOMATE_MENUS", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        max_shop_buys=max(0, int(os.environ.get("BROTATO_V3_MAX_SHOP_BUYS", "4"))),
        max_shop_rerolls=max(0, int(os.environ.get("BROTATO_V3_MAX_SHOP_REROLLS", "1"))),
        ui_build_profile=os.environ.get("BROTATO_V3_UI_BUILD_PROFILE", "stick_melee")
        .strip()
        .lower(),
        ui_model_path=ui_model_path,
        ui_decision_log=ui_decision_log,
        safety_shield=os.environ.get("BROTATO_V3_SAFETY_SHIELD", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        combat_decision_log=combat_decision_log,
    )
