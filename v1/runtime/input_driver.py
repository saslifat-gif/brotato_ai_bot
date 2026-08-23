"""Single, deterministic input boundary for the game runtime.

The driver has two explicit contracts: ``background`` Win32 messages or
``physical_foreground`` input.  An action never silently switches between
them, so a successful return value means the selected backend actually sent
the event.
"""

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
            raise RuntimeError("pydirectinput is unavailable")

        @staticmethod
        def keyDown(_key):
            raise RuntimeError("pydirectinput is unavailable")

        @staticmethod
        def keyUp(_key):
            raise RuntimeError("pydirectinput is unavailable")

        @staticmethod
        def click(_x=None, _y=None):
            raise RuntimeError("pydirectinput is unavailable")

    pydirectinput = _NoDirectInput()  # type: ignore[assignment]


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


VK_MAP = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44, "enter": 0x0D}


def normalize_input_mode(value: str) -> str:
    """Map legacy values to the two supported input contracts."""
    mode = str(value or "physical_foreground").strip().lower().replace("-", "_")
    if mode in {"safe_background", "background", "post_message"}:
        return "background"
    if mode in {"aggressive_click", "physical", "physical_click", "physical_foreground"}:
        return "physical_foreground"
    return "physical_foreground"


class InputDriver:
    def __init__(
        self,
        hwnd: int,
        input_mode: str = "physical_foreground",
        allow_physical_fallback: bool = False,
        move_physical: bool = True,
        focus_timeout_sec: float = 0.20,
    ):
        self.hwnd = int(hwnd)
        self.input_mode = normalize_input_mode(input_mode)
        self.allow_physical_fallback = bool(allow_physical_fallback)
        self.move_physical = bool(move_physical)
        self.focus_timeout_sec = max(0.02, float(focus_timeout_sec))
        self.current_move_keys: set[str] = set()
        self.last_error = ""
        try:
            pydirectinput.FAILSAFE = False
            pydirectinput.PAUSE = 0.0
        except Exception:
            pass

    def _is_game_foreground(self) -> bool:
        try:
            return int(ctypes.windll.user32.GetForegroundWindow()) == self.hwnd
        except Exception:
            return False

    def focus_game(self) -> bool:
        """Bring the game forward and verify Windows accepted the request."""
        try:
            user32 = ctypes.windll.user32
            user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(self.hwnd)
            user32.SetActiveWindow(self.hwnd)
        except Exception as exc:
            self.last_error = f"focus_exception:{exc}"
            return False
        deadline = time.perf_counter() + self.focus_timeout_sec
        while time.perf_counter() < deadline:
            if self._is_game_foreground():
                return True
            time.sleep(0.005)
        self.last_error = "focus_rejected"
        return False

    @staticmethod
    def _vk_of(key: str) -> Optional[int]:
        return VK_MAP.get(str(key or "").strip().lower())

    def _post_key(self, key: str, down: bool) -> bool:
        vk = self._vk_of(key)
        if vk is None:
            self.last_error = f"unsupported_key:{key}"
            return False
        try:
            msg = 0x0100 if down else 0x0101
            return bool(ctypes.windll.user32.PostMessageW(self.hwnd, msg, vk, 0))
        except Exception as exc:
            self.last_error = f"post_key_exception:{exc}"
            return False

    def _is_client_pos_valid(self, pos: Tuple[int, int]) -> bool:
        try:
            crect = RECT()
            if not ctypes.windll.user32.GetClientRect(self.hwnd, ctypes.byref(crect)):
                return False
            x, y = int(pos[0]), int(pos[1])
            return int(crect.right) > 0 and int(crect.bottom) > 0 and 0 <= x < int(crect.right) and 0 <= y < int(crect.bottom)
        except Exception:
            return False

    def _client_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        pt = POINT(int(pos[0]), int(pos[1]))
        if not ctypes.windll.user32.ClientToScreen(self.hwnd, ctypes.byref(pt)):
            raise RuntimeError("ClientToScreen failed")
        return int(pt.x), int(pt.y)

    @staticmethod
    def _cursor_pos() -> Optional[Tuple[int, int]]:
        try:
            pt = POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
        except Exception:
            pass
        return None

    @staticmethod
    def _restore_cursor(pos: Optional[Tuple[int, int]]) -> None:
        if pos is None:
            return
        try:
            ctypes.windll.user32.SetCursorPos(int(pos[0]), int(pos[1]))
        except Exception:
            pass

    def _background_click(self, x: int, y: int) -> ClickResult:
        if not self._is_client_pos_valid((x, y)):
            return ClickResult(False, "background", "invalid_client_point")
        try:
            lparam = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
            user32 = ctypes.windll.user32
            down = bool(user32.PostMessageW(self.hwnd, 0x0201, 0x0001, lparam))
            time.sleep(0.025)
            up = bool(user32.PostMessageW(self.hwnd, 0x0202, 0x0000, lparam))
            if down and up:
                return ClickResult(True, "background")
            return ClickResult(False, "background", "post_message_rejected")
        except Exception as exc:
            return ClickResult(False, "background", f"post_message_exception:{exc}")

    def _physical_click(self, x: int, y: int) -> ClickResult:
        if not self.focus_game():
            return ClickResult(False, "physical_foreground", self.last_error or "focus_rejected")
        cursor = self._cursor_pos()
        try:
            sx, sy = self._client_to_screen((x, y))
            pydirectinput.click(sx, sy)
            return ClickResult(True, "physical_foreground")
        except Exception as exc:
            return ClickResult(False, "physical_foreground", str(exc))
        finally:
            self._restore_cursor(cursor)

    def click_client_point(self, pos: Tuple[int, int]) -> ClickResult:
        x, y = int(pos[0]), int(pos[1])
        if not self._is_client_pos_valid((x, y)):
            return ClickResult(False, self.input_mode, "invalid_client_point")
        if self.input_mode == "background":
            return self._background_click(x, y)
        return self._physical_click(x, y)

    def click_client_rect(self, rect: Tuple[int, int, int, int]) -> ClickResult:
        x1, y1, x2, y2 = [int(v) for v in rect]
        lx, rx = sorted((x1, x2))
        ty, by = sorted((y1, y2))
        px = lx if rx <= lx else random.randint(lx, rx - 1)
        py = ty if by <= ty else random.randint(ty, by - 1)
        return self.click_client_point((px, py))

    def press_key(self, key: str) -> bool:
        k = str(key or "").strip().lower()
        if self.input_mode == "background":
            return self._post_key(k, True) and self._post_key(k, False)
        if not self.focus_game():
            return False
        try:
            pydirectinput.press(k)
            return True
        except Exception as exc:
            self.last_error = f"physical_key_exception:{exc}"
            return False

    def _hold_key_down(self, key: str) -> bool:
        if self.input_mode == "background":
            return self._post_key(key, True)
        if not self.focus_game():
            return False
        try:
            pydirectinput.keyDown(key)
            return True
        except Exception as exc:
            self.last_error = f"physical_keydown_exception:{exc}"
            return False

    def _hold_key_up(self, key: str) -> bool:
        if self.input_mode == "background":
            return self._post_key(key, False)
        try:
            pydirectinput.keyUp(key)
            return True
        except Exception as exc:
            self.last_error = f"physical_keyup_exception:{exc}"
            return False

    def set_move_keys(self, move_keys) -> None:
        desired = set() if move_keys is None else {
            str(k).strip().lower() for k in move_keys if str(k).strip().lower() in {"w", "a", "s", "d"}
        }
        for key in sorted(self.current_move_keys - desired):
            self._hold_key_up(key)
            self.current_move_keys.discard(key)
        for key in sorted(desired - self.current_move_keys):
            if self._hold_key_down(key):
                self.current_move_keys.add(key)

    def set_move_key(self, move_key: Optional[str]) -> None:
        self.set_move_keys(None if move_key is None else [move_key])

    def release_movement(self) -> None:
        self.set_move_keys(None)
