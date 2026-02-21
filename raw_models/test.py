import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure the `v1` package folder is on sys.path so `env` can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v1"))

from env.roboflow_detector import RoboflowMonsterDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_unicode(path: Path):
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"[error] failed to read {path}: {exc}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Roboflow detector on a folder of images.")
    parser.add_argument(
        "--images-dir",
        default=os.environ.get("BROTATO_TEST_IMAGES_DIR", ""),
        help="Input image directory (or set BROTATO_TEST_IMAGES_DIR).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")

    images_dir = Path(args.images_dir).expanduser()
    if not str(images_dir):
        raise RuntimeError("Missing --images-dir (or BROTATO_TEST_IMAGES_DIR).")
    if not images_dir.exists() or not images_dir.is_dir():
        raise RuntimeError(f"images dir not found: {images_dir}")

    detector = RoboflowMonsterDetector(api_key=api_key)

    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = imread_unicode(path)
        if img is None:
            print(f"[warn] skip unreadable: {path}")
            continue
        print(f"[run] {path}")
        detector.print_classes(img)


if __name__ == "__main__":
    main()
