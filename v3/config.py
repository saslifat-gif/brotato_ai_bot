"""Configuration for the API-first trainer."""

import os
from dataclasses import dataclass
from pathlib import Path

from v3.protocol import DEFAULT_HOST, DEFAULT_PORT


@dataclass(frozen=True)
class V3Config:
    host: str
    port: int
    state_timeout_sec: float
    reset_timeout_sec: float
    output_dir: Path
    total_timesteps: int


def load_config() -> V3Config:
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("BROTATO_V3_OUTPUT_DIR", root / "models" / "version_3"))
    if not output.is_absolute():
        output = (root / output).resolve()
    return V3Config(
        host=os.environ.get("BROTATO_V3_HOST", DEFAULT_HOST),
        port=int(os.environ.get("BROTATO_V3_PORT", str(DEFAULT_PORT))),
        state_timeout_sec=max(1.0, float(os.environ.get("BROTATO_V3_STATE_TIMEOUT", "30"))),
        reset_timeout_sec=max(10.0, float(os.environ.get("BROTATO_V3_RESET_TIMEOUT", "600"))),
        output_dir=output,
        total_timesteps=max(1, int(os.environ.get("BROTATO_V3_TIMESTEPS", "1000000"))),
    )
