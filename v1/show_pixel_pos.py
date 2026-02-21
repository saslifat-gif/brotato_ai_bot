import ctypes
import time

import cv2
import numpy as np

try:
    import windows_capture as wc
except Exception as e:
    raise ImportError("Please install windows-capture: pip install windows-capture") from e


WINDOW_TITLE = "Brotato"
EXE_NAME = "Brotato.exe"
PREVIEW_SCALE = 1.25


class WindowsCaptureAdapter:
    def __init__(self, window_name: str):
        self._latest_bgr = None
        self._control = None
        self._finished = False
        self._cap = wc.WindowsCapture(window_name=window_name)

        @self._cap.event
        def on_frame_arrived(frame, control):
            if self._control is None:
                self._control = control
            self._latest_bgr = np.ascontiguousarray(frame.convert_to_bgr().frame_buffer)

        @self._cap.event
        def on_closed():
            self._finished = True

        self._control = self._cap.start_free_threaded()

    def get_latest_bgr(self):
        return self._latest_bgr

    def stop(self):
        try:
            if self._control is not None and not self._control.is_finished():
                self._control.stop()
        except Exception:
            pass


def find_hwnd_by_exe(exe_name: str):
    hwnd_found = None
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_long)

    def enum_cb(hwnd, _):
        nonlocal hwnd_found
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        h_process = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
        if h_process:
            buf = ctypes.create_unicode_buffer(1024)
            try:
                ctypes.windll.psapi.GetModuleBaseNameW(h_process, None, buf, 1024)
                if buf.value.lower() == exe_name.lower():
                    hwnd_found = hwnd
                    ctypes.windll.kernel32.CloseHandle(h_process)
                    return False
            except Exception:
                pass
            ctypes.windll.kernel32.CloseHandle(h_process)
        return True

    ctypes.windll.user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    return hwnd_found


def resolve_window_title(title: str, exe_name: str) -> str:
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if hwnd:
        return title

    hwnd = find_hwnd_by_exe(exe_name)
    if not hwnd:
        raise RuntimeError(f"Cannot find game window by title='{title}' or exe='{exe_name}'")

    n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
    out = buf.value.strip()
    if not out:
        raise RuntimeError("Found hwnd but title is empty")
    return out


def main():
    title = resolve_window_title(WINDOW_TITLE, EXE_NAME)
    print(f"[capture] backend=windows-capture, window='{title}'")
    print("[tips] Move mouse on preview window to inspect coords. Left click to print. Press 'q' to quit.")

    cam = WindowsCaptureAdapter(title)
    time.sleep(0.3)

    state = {
        "mx": -1,
        "my": -1,
        "click": None,
    }

    def on_mouse(event, x, y, flags, param):
        state["mx"] = x
        state["my"] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow("Pixel Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pixel Inspector", on_mouse)

    while True:
        frame = cam.get_latest_bgr()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]
        disp_w = int(w * PREVIEW_SCALE)
        disp_h = int(h * PREVIEW_SCALE)
        disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        mx, my = state["mx"], state["my"]
        if mx >= 0 and my >= 0:
            x = int(round(mx / PREVIEW_SCALE))
            y = int(round(my / PREVIEW_SCALE))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))

            b, g, r = frame[y, x].tolist()

            cv2.line(disp, (mx, 0), (mx, disp_h - 1), (0, 255, 255), 1)
            cv2.line(disp, (0, my), (disp_w - 1, my), (0, 255, 255), 1)
            cv2.circle(disp, (mx, my), 4, (0, 255, 0), -1)

            text1 = f"window_xy=({x},{y})"
            text2 = f"BGR=({b},{g},{r})"
            cv2.rectangle(disp, (8, 8), (340, 56), (0, 0, 0), -1)
            cv2.putText(disp, text1, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(disp, text2, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if state["click"] is not None:
            cx, cy = state["click"]
            x = int(round(cx / PREVIEW_SCALE))
            y = int(round(cy / PREVIEW_SCALE))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            b, g, r = frame[y, x].tolist()
            print(f"click window_xy=({x},{y}) BGR=({b},{g},{r})")
            state["click"] = None

        cv2.imshow("Pixel Inspector", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
