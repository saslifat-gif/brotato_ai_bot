"""Minimal Win32 game-window discovery used by recording and training."""

import ctypes
from typing import Optional, Tuple

from v1.runtime.input_driver import POINT, RECT, enable_process_dpi_awareness


enable_process_dpi_awareness()


def _find_window_fuzzy(keyword: str) -> Optional[int]:
    found = None
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_long)

    def visit(hwnd, _param):
        nonlocal found
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        text = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, text, length + 1)
        if str(keyword).lower() in text.value.lower():
            found = int(hwnd)
            return False
        return True

    ctypes.windll.user32.EnumWindows(enum_proc(visit), 0)
    return found


def find_game_window(title: str = "Brotato") -> int:
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, str(title))
        if not hwnd:
            hwnd = _find_window_fuzzy(title)
    except Exception as exc:
        raise RuntimeError("v2 runtime requires Windows") from exc
    if not hwnd:
        raise RuntimeError(f"game window not found: {title!r}")
    return int(hwnd)


def client_screen_rect(hwnd: int) -> Tuple[int, int, int, int]:
    rect = RECT()
    if not ctypes.windll.user32.GetClientRect(int(hwnd), ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")
    top_left = POINT(int(rect.left), int(rect.top))
    bottom_right = POINT(int(rect.right), int(rect.bottom))
    if not ctypes.windll.user32.ClientToScreen(int(hwnd), ctypes.byref(top_left)):
        raise RuntimeError("ClientToScreen(top-left) failed")
    if not ctypes.windll.user32.ClientToScreen(int(hwnd), ctypes.byref(bottom_right)):
        raise RuntimeError("ClientToScreen(bottom-right) failed")
    return int(top_left.x), int(top_left.y), int(bottom_right.x), int(bottom_right.y)
