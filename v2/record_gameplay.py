"""Record gameplay frames and the human WASD action for v2 training data."""

import argparse
import csv
import ctypes
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from v1.runtime.capture import create_camera
from v2.config import load_config
from v2.runtime.window import client_screen_rect, find_game_window


VK = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44, "stop": 0x77}
ACTION_FOR_KEYS = {
    frozenset(): 0,
    frozenset({"w"}): 1,
    frozenset({"s"}): 2,
    frozenset({"a"}): 3,
    frozenset({"d"}): 4,
    frozenset({"w", "a"}): 5,
    frozenset({"w", "d"}): 6,
    frozenset({"s", "a"}): 7,
    frozenset({"s", "d"}): 8,
}


def _pressed(vk: int) -> bool:
    return bool(int(ctypes.windll.user32.GetAsyncKeyState(int(vk))) & 0x8000)


def current_action() -> tuple[int, str]:
    keys = frozenset(key for key in ("w", "a", "s", "d") if _pressed(VK[key]))
    return ACTION_FOR_KEYS.get(keys, 0), "+".join(sorted(keys))


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Brotato gameplay for v2 perception/imitation data")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--output", default="datasets/v2/raw")
    args = parser.parse_args()

    cfg = load_config()
    hwnd = find_game_window(cfg.window_title)
    region = client_screen_rect(hwnd)
    camera = create_camera(cfg.capture_backend, region, target_fps=max(30, int(args.fps * 2)))

    root = Path(args.output).resolve() / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    actions_path = root / "actions.csv"
    interval = 1.0 / max(1.0, float(args.fps))
    quality = int(max(50, min(100, args.jpeg_quality)))

    print(f"[record] output={root}")
    print(f"[record] region={region} fps={args.fps:.1f}; press F8 to stop")
    started = time.time()
    frame_id = 0
    try:
        with actions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("frame_id", "timestamp_sec", "image", "action", "keys"),
            )
            writer.writeheader()
            while not _pressed(VK["stop"]):
                tick = time.perf_counter()
                frame = camera.get_latest_frame()
                if frame is not None and frame.size > 0:
                    action, keys = current_action()
                    image_name = f"frame_{frame_id:08d}.jpg"
                    cv2.imwrite(
                        str(frames_dir / image_name),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, quality],
                    )
                    writer.writerow(
                        {
                            "frame_id": frame_id,
                            "timestamp_sec": f"{time.time() - started:.6f}",
                            "image": f"frames/{image_name}",
                            "action": action,
                            "keys": keys,
                        }
                    )
                    frame_id += 1
                remaining = interval - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        metadata = {
            "window_title": cfg.window_title,
            "region": region,
            "fps": float(args.fps),
            "frames": frame_id,
            "duration_sec": time.time() - started,
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[record] saved frames={frame_id} actions={actions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

