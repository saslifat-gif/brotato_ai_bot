import ctypes
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import pydirectinput
except Exception:
    class _NoDirectInput:
        FAILSAFE = False
        PAUSE = 0.0

        @staticmethod
        def press(_key):
            return None

        @staticmethod
        def keyDown(_key):
            return None

        @staticmethod
        def keyUp(_key):
            return None

        @staticmethod
        def click(_x=None, _y=None):
            return None

    pydirectinput = _NoDirectInput()  # type: ignore[assignment]
    print("[input] pydirectinput unavailable; using no-op fallback")


@dataclass
class ClickResult:
    ok: bool
    method: str
    error: str = ""


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


VK_MAP = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "enter": 0x0D,
}


class InputDriver:
    def __init__(
        self,
        hwnd: int,
        input_mode: str = "safe_background",
        allow_physical_fallback: bool = False,
        move_physical: bool = True,
    ):
        self.hwnd = hwnd
        self.input_mode = str(input_mode or "safe_background").strip().lower()
        self.allow_physical_fallback = bool(allow_physical_fallback)
        self.move_physical = bool(move_physical)
        self.current_move_key: Optional[str] = None

        pydirectinput.FAILSAFE = False
        try:
            pydirectinput.PAUSE = 0.0
        except Exception:
            pass

    def _is_game_foreground(self) -> bool:
        try:
            fg = int(ctypes.windll.user32.GetForegroundWindow())
            return fg == int(self.hwnd)
        except Exception:
            return False

    def _focus_if_needed(self):
        if self.input_mode != "aggressive_click":
            return
        try:
            ctypes.windll.user32.ShowWindow(self.hwnd, 5)
            ctypes.windll.user32.SetForegroundWindow(self.hwnd)
            ctypes.windll.user32.SetActiveWindow(self.hwnd)
        except Exception:
            pass

    @staticmethod
    def _vk_of(key: str) -> Optional[int]:
        k = str(key or "").strip().lower()
        return VK_MAP.get(k)

    def _post_key_down(self, key: str) -> bool:
        vk = self._vk_of(key)
        if vk is None:
            return False
        try:
            ctypes.windll.user32.PostMessageW(self.hwnd, 0x0100, vk, 0)
            return True
        except Exception:
            return False

    def _post_key_up(self, key: str) -> bool:
        vk = self._vk_of(key)
        if vk is None:
            return False
        try:
            ctypes.windll.user32.PostMessageW(self.hwnd, 0x0101, vk, 0)
            return True
        except Exception:
            return False

    def _is_client_pos_valid(self, pos: Tuple[int, int]) -> bool:
        x, y = int(pos[0]), int(pos[1])
        crect = RECT()
        if not ctypes.windll.user32.GetClientRect(self.hwnd, ctypes.byref(crect)):
            return False
        cw, ch = int(crect.right), int(crect.bottom)
        return cw > 0 and ch > 0 and (0 <= x < cw) and (0 <= y < ch)

    def _client_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        pt = POINT()
        pt.x = int(pos[0])
        pt.y = int(pos[1])
        ctypes.windll.user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def click_client_point(self, pos: Tuple[int, int]) -> ClickResult:
        self._focus_if_needed()
        x, y = int(pos[0]), int(pos[1])
        try:
            if self._is_client_pos_valid((x, y)):
                lparam = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
                ctypes.windll.user32.PostMessageW(self.hwnd, 0x0201, 0x0001, lparam)
                time.sleep(0.025)
                ctypes.windll.user32.PostMessageW(self.hwnd, 0x0202, 0x0000, lparam)
                return ClickResult(ok=True, method="post_message")
        except Exception as e:
            post_error = str(e)
        else:
            post_error = "post_message_failed"

        if not self.allow_physical_fallback:
            return ClickResult(ok=False, method="post_message", error=post_error)

        try:
            sx, sy = self._client_to_screen((x, y))
            pydirectinput.click(sx, sy)
            return ClickResult(ok=True, method="physical")
        except Exception as e:
            return ClickResult(ok=False, method="physical", error=str(e))

    def click_client_rect(self, rect: Tuple[int, int, int, int]) -> ClickResult:
        x1, y1, x2, y2 = rect
        lx, rx = sorted((int(x1), int(x2)))
        ty, by = sorted((int(y1), int(y2)))
        if lx == rx and ty == by:
            return self.click_client_point((lx, ty))
        px = int(random.randint(lx, rx))
        py = int(random.randint(ty, by))
        return self.click_client_point((px, py))

    def press_key(self, key: str):
        self._focus_if_needed()
        k = str(key).strip().lower()
        sent = self._post_key_down(k)
        time.sleep(0.005)
        sent = self._post_key_up(k) and sent
        if sent:
            return
        if not self.allow_physical_fallback:
            return
        try:
            pydirectinput.press(k)
        except Exception:
            pass

    def set_move_key(self, move_key: Optional[str]):
        key = None if move_key is None else str(move_key).strip().lower()
        if key == self.current_move_key:
            return
        if self.current_move_key is not None:
            if self.move_physical:
                try:
                    pydirectinput.keyUp(self.current_move_key)
                except Exception:
                    pass
            else:
                if not self._post_key_up(self.current_move_key):
                    if self.allow_physical_fallback:
                        try:
                            pydirectinput.keyUp(self.current_move_key)
                        except Exception:
                            pass
            self.current_move_key = None
        if key in ("w", "a", "s", "d"):
            if self.move_physical:
                if not self._is_game_foreground():
                    return
                try:
                    pydirectinput.keyDown(key)
                    self.current_move_key = key
                except Exception:
                    self.current_move_key = None
            else:
                if self._post_key_down(key):
                    self.current_move_key = key
                elif self.allow_physical_fallback:
                    try:
                        pydirectinput.keyDown(key)
                        self.current_move_key = key
                    except Exception:
                        self.current_move_key = None

    def release_movement(self):
        self.set_move_key(None)
