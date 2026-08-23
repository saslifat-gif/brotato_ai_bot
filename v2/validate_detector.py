"""Live visual validation for custom v2 detector weights."""

import argparse
import ctypes
import time

import cv2

from v1.runtime.capture import create_camera
from v2.config import load_config
from v2.perception.yolo_detector import YoloDetector
from v2.runtime.window import client_screen_rect, find_game_window


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("combat", "ui"), default="combat")
    args = parser.parse_args()
    cfg = load_config()
    weights = cfg.combat_weights if args.task == "combat" else cfg.ui_weights
    detector = YoloDetector(
        str(weights),
        confidence=cfg.detector_confidence,
        image_size=cfg.detector_image_size,
        device=cfg.detector_device,
    )
    hwnd = find_game_window(cfg.window_title)
    region = client_screen_rect(hwnd)
    camera = create_camera(cfg.capture_backend, region, target_fps=60)
    print(f"[v2-validate] task={args.task} weights={weights}; press F8 or Esc to stop")
    try:
        while True:
            started = time.perf_counter()
            frame = camera.get_latest_frame()
            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue
            result = detector.detect(frame, track=args.task == "combat")
            canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for item in result.items:
                x1, y1, x2, y2 = [int(round(v)) for v in item.box]
                color = (0, 220, 0) if item.label == "player" else (0, 160, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                text = f"{item.label} {item.confidence:.2f}"
                cv2.putText(canvas, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            fps = 1.0 / max(1e-6, time.perf_counter() - started)
            cv2.putText(canvas, f"{args.task} {fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Brotato v2 Detector Validation", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or bool(ctypes.windll.user32.GetAsyncKeyState(0x77) & 0x8000):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

