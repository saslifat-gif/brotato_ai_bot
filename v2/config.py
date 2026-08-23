import os
from dataclasses import dataclass
from pathlib import Path


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class V2Config:
    window_title: str
    capture_backend: str
    combat_weights: Path
    ui_weights: Path
    detector_device: str
    detector_image_size: int
    detector_confidence: float
    action_interval_sec: float
    output_dir: Path
    require_detector_weights: bool


def load_config() -> V2Config:
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(os.environ.get("BROTATO_V2_OUTPUT_DIR", root / "models" / "version_2"))
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    combat_weights = Path(os.environ.get("BROTATO_V2_COMBAT_WEIGHTS", output_dir / "combat_best.pt"))
    ui_weights = Path(os.environ.get("BROTATO_V2_UI_WEIGHTS", output_dir / "ui_best.pt"))
    if not combat_weights.is_absolute():
        combat_weights = (root / combat_weights).resolve()
    if not ui_weights.is_absolute():
        ui_weights = (root / ui_weights).resolve()
    return V2Config(
        window_title=os.environ.get("BROTATO_WINDOW_TITLE", "Brotato"),
        capture_backend=os.environ.get("BROTATO_CAPTURE_BACKEND", "windows-capture").strip().lower(),
        combat_weights=combat_weights,
        ui_weights=ui_weights,
        detector_device=os.environ.get("BROTATO_V2_DEVICE", "cpu"),
        detector_image_size=max(160, int(os.environ.get("BROTATO_V2_IMGSZ", "416"))),
        detector_confidence=float(os.environ.get("BROTATO_V2_CONF", "0.25")),
        action_interval_sec=max(0.03, float(os.environ.get("BROTATO_V2_ACTION_INTERVAL", "0.08"))),
        output_dir=output_dir,
        require_detector_weights=_boolean("BROTATO_V2_REQUIRE_WEIGHTS", True),
    )
