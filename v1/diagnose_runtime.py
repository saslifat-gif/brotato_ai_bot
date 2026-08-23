"""Preflight checks for the game window, capture region and input contract.

This deliberately does not start PPO or click the shop.  Run it before a
training session to distinguish a bad window/capture/input setup from a model
problem.  ``--move-test`` sends a short W key hold only when explicitly asked.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.runtime_config import load_runtime_config  # noqa: E402
from env.brotato_env import force_game_window, hwnd_client_screen_rect, monitor_for_point  # noqa: E402
from runtime.capture import create_camera  # noqa: E402
from runtime.input_driver import DPI_AWARENESS, InputDriver  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Brotato runtime preflight")
    parser.add_argument("--focus", action="store_true", help="bring the game window to the foreground")
    parser.add_argument("--move-test", action="store_true", help="hold W for 0.35 seconds")
    args = parser.parse_args()

    cfg = load_runtime_config()
    hwnd = force_game_window(cfg.window_title, cfg.exe_name, cfg.force_resize, cfg.window_w, cfg.window_h)
    if not hwnd:
        print(f"[diagnose] game window not found title={cfg.window_title!r} exe={cfg.exe_name!r}")
        return 2

    region = hwnd_client_screen_rect(hwnd)
    cx = (region[0] + region[2]) // 2
    cy = (region[1] + region[3]) // 2
    monitor_index, monitor_origin = monitor_for_point(cx, cy)
    print(f"[diagnose] hwnd={hwnd} region={region} monitor_index={monitor_index} origin={monitor_origin}")
    print(f"[diagnose] capture={cfg.capture_backend} input={cfg.input_mode} control_panel={getattr(cfg, 'control_panel_enabled', False)}")
    print(f"[diagnose] dpi_awareness={DPI_AWARENESS}")

    driver = InputDriver(hwnd, cfg.input_mode, cfg.input_physical_fallback, cfg.input_move_physical)
    if args.focus or args.move_test:
        print(f"[diagnose] focus={'ok' if driver.focus_game() else 'failed'} error={driver.last_error}")
    if args.move_test:
        driver.set_move_key("w")
        time.sleep(0.35)
        driver.release_movement()
        print(f"[diagnose] movement keys released; input_error={driver.last_error}")

    camera = None
    try:
        camera = create_camera(cfg.capture_backend, region, monitor_index, monitor_origin, target_fps=30)
        deadline = time.time() + 2.0
        frame = None
        while time.time() < deadline and frame is None:
            frame = camera.get_latest_frame()
            time.sleep(0.05)
        if frame is None:
            print("[diagnose] capture failed: no frame received")
            return 3
        print(f"[diagnose] capture ok backend={getattr(camera, 'backend_name', cfg.capture_backend)} frame_shape={frame.shape}")
    except Exception as exc:
        print(f"[diagnose] capture failed: {exc}")
        return 3
    finally:
        if camera is not None:
            camera.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
