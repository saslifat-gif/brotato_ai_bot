import threading
import time
from typing import Dict

import cv2
import numpy as np


DEBUG_WINDOW_HP = "HP Crop Debug"
DEBUG_WINDOW_RED_GREEN = "RedGreen Map"
DEBUG_WINDOW_SHOP = "Shop Debug"
DEBUG_WINDOW_STATE = "State Debug"

CORE_DEBUG_WINDOWS = (
    DEBUG_WINDOW_HP,
    DEBUG_WINDOW_RED_GREEN,
    DEBUG_WINDOW_SHOP,
    DEBUG_WINDOW_STATE,
)


class DebugWindowManager:
    def __init__(self, enabled: bool = True, target_fps: int = 12):
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._target_fps = max(1, int(target_fps))
        self._interval = 1.0 / float(self._target_fps)
        self._last_render_ts = 0.0

    def toggle(self) -> bool:
        with self._lock:
            self.enabled = not self.enabled
            enabled = self.enabled
        if not enabled:
            self.close_all()
        return enabled

    def set_enabled(self, enabled: bool):
        with self._lock:
            self.enabled = bool(enabled)
        if not enabled:
            self.close_all()

    def can_render(self) -> bool:
        with self._lock:
            if not self.enabled:
                return False
        now = time.perf_counter()
        if now - self._last_render_ts < self._interval:
            return False
        self._last_render_ts = now
        return True

    def show(self, window_name: str, image: np.ndarray):
        with self._lock:
            if not self.enabled:
                return
        if window_name not in CORE_DEBUG_WINDOWS:
            return
        try:
            cv2.imshow(window_name, image)
        except Exception:
            pass

    def show_many(self, frames: Dict[str, np.ndarray]):
        with self._lock:
            if not self.enabled:
                return
        for name in CORE_DEBUG_WINDOWS:
            img = frames.get(name)
            if img is None:
                continue
            try:
                cv2.imshow(name, img)
            except Exception:
                pass
        try:
            cv2.waitKey(1)
        except Exception:
            pass

    def close_all(self):
        for name in CORE_DEBUG_WINDOWS:
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
        try:
            cv2.waitKey(1)
        except Exception:
            pass

