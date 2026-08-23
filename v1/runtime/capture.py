"""Capture boundary with monitor-independent coordinates.

MSS accepts absolute virtual-desktop coordinates, which avoids the common
Windows-capture monitor-index mismatch on two-monitor setups.  The optional
Windows Capture backend remains available for users who explicitly select it.
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    import mss
except Exception:
    mss = None

try:
    import windows_capture as wc
except Exception:
    wc = None


class MSSCamera:
    backend_name = "mss"

    def __init__(self, region: Tuple[int, int, int, int], target_fps: int = 90):
        if mss is None:
            raise RuntimeError("mss is not installed")
        self.region = tuple(int(v) for v in region)
        self.interval = 1.0 / max(1, int(target_fps))
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="brotato-capture", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        left, top, right, bottom = self.region
        monitor = {
            "left": left,
            "top": top,
            "width": max(1, right - left),
            "height": max(1, bottom - top),
        }
        try:
            with mss.mss() as screen:
                while self._running:
                    started = time.perf_counter()
                    raw = np.asarray(screen.grab(monitor))
                    # mss returns BGRA.  The environment consumes RGB.
                    frame = np.ascontiguousarray(raw[:, :, [2, 1, 0]])
                    with self._lock:
                        self._latest = frame
                    remaining = self.interval - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        except Exception as exc:
            # Keep the failure visible to the runtime rather than silently
            # switching to a different monitor/backend mid-episode.
            self.error = str(exc)
            self._running = False

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class WindowsCaptureCamera:
    backend_name = "windows-capture"

    def __init__(self, monitor_index: int, region_in_monitor: Tuple[int, int, int, int]):
        if wc is None:
            raise RuntimeError("windows-capture is not installed")
        self._lock = threading.Lock()
        self._latest = None
        self._region = tuple(int(v) for v in region_in_monitor)
        self._control = None
        self._cap = wc.WindowsCapture(monitor_index=int(monitor_index) + 1)

        @self._cap.event
        def on_frame_arrived(frame, control):
            if self._control is None:
                self._control = control
            fb = frame.convert_to_bgr().frame_buffer
            x1, y1, x2, y2 = self._region
            h, w = fb.shape[:2]
            x1 = int(np.clip(x1, 0, max(0, w - 1)))
            y1 = int(np.clip(y1, 0, max(0, h - 1)))
            x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
            y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
            out = fb[y1:y2, x1:x2]
            with self._lock:
                self._latest = np.ascontiguousarray(out[:, :, ::-1])

        @self._cap.event
        def on_closed():
            return None

        self._control = self._cap.start_free_threaded()

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        try:
            if self._control is not None and not self._control.is_finished():
                self._control.stop()
        except Exception:
            pass


def create_camera(
    backend: str,
    region: Tuple[int, int, int, int],
    monitor_index: int = 0,
    monitor_origin: Tuple[int, int] = (0, 0),
    target_fps: int = 90,
):
    """Create exactly one backend; ``auto`` prefers MSS for multi-monitor safety."""
    requested = str(backend or "mss").strip().lower()
    if requested in {"mss", "auto"}:
        return MSSCamera(region, target_fps=target_fps).start()
    if requested == "windows-capture":
        left, top, _, _ = region
        ox, oy = monitor_origin
        return WindowsCaptureCamera(
            monitor_index=monitor_index,
            region_in_monitor=(left - ox, top - oy, region[2] - ox, region[3] - oy),
        )
    raise ValueError(f"unsupported capture backend: {backend}")
