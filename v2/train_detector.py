"""Train the custom YOLO26n combat or UI detector."""

import argparse
import shutil
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("combat", "ui"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")

    root = Path(__file__).resolve().parents[1]
    data = root / "v2" / "data" / f"{args.task}.yaml"
    if not data.exists():
        raise RuntimeError(f"dataset config missing: {data}")
    dataset_root = (root / "datasets" / "v2" / args.task).resolve()
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise RuntimeError("dataset is not ready; missing:\n" + "\n".join(missing))
    dataset_cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    dataset_cfg["path"] = str(dataset_root)
    resolved_data = root / "models" / "version_2" / f"{args.task}_dataset.resolved.yaml"
    resolved_data.parent.mkdir(parents=True, exist_ok=True)
    resolved_data.write_text(yaml.safe_dump(dataset_cfg, sort_keys=False), encoding="utf-8")
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(resolved_data),
        epochs=max(1, int(args.epochs)),
        imgsz=max(160, int(args.imgsz)),
        device=str(args.device),
        project=str(root / "models" / "version_2" / "detector_runs"),
        name=args.task,
    )
    best = Path(str(model.trainer.best))
    destination = root / "models" / "version_2" / f"{args.task}_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"[v2-detector] best weights copied to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
