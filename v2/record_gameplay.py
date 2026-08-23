"""Record gameplay frames and the human WASD action for v2 training data."""

import argparse
import csv
import ctypes
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import msvcrt
except ImportError:  # pragma: no cover - the recorder itself is Windows-only
    msvcrt = None

from v1.runtime.capture import create_camera
from v1.runtime.input_driver import InputDriver
from v2.config import load_config
from v2.runtime.window import client_screen_rect, find_game_window, monitor_for_region


VK = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44}
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


def console_stop_requested() -> bool:
    """Stop only from the focused recorder console, never from inside the game."""
    if msvcrt is None or not msvcrt.kbhit():
        return False
    key = msvcrt.getwch()
    return key in {"q", "Q", "\r", "\n"}


def visual_change_score(first: np.ndarray, second: np.ndarray) -> float:
    """Measure frame-to-frame motion on a small grayscale preview."""
    def preview(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)

    return float(np.mean(cv2.absdiff(preview(first), preview(second))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Brotato gameplay for v2 perception/imitation data")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--output", default="datasets/v2/raw")
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    hwnd = find_game_window(cfg.window_title)
    region = client_screen_rect(hwnd)
    monitor_index, monitor_origin = monitor_for_region(region)
    camera = create_camera(
        cfg.capture_backend,
        region,
        monitor_index=monitor_index,
        monitor_origin=monitor_origin,
        target_fps=max(30, int(args.fps * 2)),
    )

    root = Path(args.output).resolve() / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    actions_path = root / "actions.csv"
    interval = 1.0 / max(1.0, float(args.fps))
    quality = int(max(50, min(100, args.jpeg_quality)))

    print(f"[record] output={root}")
    backend_name = str(getattr(camera, "backend_name", cfg.capture_backend))
    print(
        f"[record] region={region} monitor={monitor_index} "
        f"backend={backend_name} fps={args.fps:.1f}"
    )
    focused = InputDriver(hwnd, input_mode="physical_foreground").focus_game()
    print(f"[record] game focus={'ok' if focused else 'not confirmed'}")
    for remaining in range(max(0, int(args.countdown)), 0, -1):
        print(f"[record] begins in {remaining}...")
        time.sleep(1.0)
    print("[record] recording; to stop, Alt+Tab here and press Q or Enter")
    started = time.time()
    frame_id = 0
    previous_frame = None
    compared_frames = 0
    changed_frames = 0
    try:
        with actions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("frame_id", "timestamp_sec", "image", "action", "keys"),
            )
            writer.writeheader()
            while not console_stop_requested():
                tick = time.perf_counter()
                frame = camera.get_latest_frame()
                if frame is not None and frame.size > 0:
                    if previous_frame is not None:
                        compared_frames += 1
                        if visual_change_score(previous_frame, frame) >= 0.5:
                            changed_frames += 1
                    previous_frame = frame
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
            "monitor_index": monitor_index,
            "capture_backend": backend_name,
            "fps": float(args.fps),
            "frames": frame_id,
            "duration_sec": time.time() - started,
            "visual_change_ratio": changed_frames / max(1, compared_frames),
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    change_ratio = changed_frames / max(1, compared_frames)
    print(f"[record] saved frames={frame_id} visual_change_ratio={change_ratio:.3f} actions={actions_path}")
    if compared_frames >= 20 and change_ratio < 0.02:
        print("[record] WARNING: capture appears frozen; do not label this session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
