import ctypes
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import windows_capture as wc
except Exception as e:
    raise ImportError("Please install windows-capture: pip install windows-capture") from e


# Relative to game window client area (left, top, right, bottom)
HP_RECT = (23, 22, 342, 70)
WINDOW_TITLE = "Brotato"
EXE_NAME = "Brotato.exe"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEALTH_ROOT = PROJECT_ROOT / "health"


class HPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 40, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


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
            buffer = ctypes.create_unicode_buffer(1024)
            try:
                ctypes.windll.psapi.GetModuleBaseNameW(h_process, None, buffer, 1024)
                if buffer.value.lower() == exe_name.lower():
                    hwnd_found = hwnd
                    ctypes.windll.kernel32.CloseHandle(h_process)
                    return False
            except Exception:
                pass
            ctypes.windll.kernel32.CloseHandle(h_process)
        return True

    ctypes.windll.user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    return hwnd_found


def find_window_title(window_title: str, exe_name: str) -> str:
    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd:
        return window_title

    hwnd = find_hwnd_by_exe(exe_name)
    if not hwnd:
        raise RuntimeError(f"Cannot find game window by title='{window_title}' or exe='{exe_name}'")

    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    resolved = buf.value.strip()
    if not resolved:
        raise RuntimeError("Found window handle by exe but title is empty")
    return resolved


class WindowsCaptureAdapter:
    def __init__(self, window_name: str):
        self._latest_bgr = None
        self._finished = False
        self._control = None
        self._cap = wc.WindowsCapture(window_name=window_name)

        @self._cap.event
        def on_frame_arrived(frame, control):
            if self._control is None:
                self._control = control
            fb = frame.convert_to_bgr().frame_buffer
            self._latest_bgr = np.ascontiguousarray(fb)

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


def crop_hp(frame_bgr: np.ndarray, rect):
    x1, y1, x2, y2 = rect
    h, w = frame_bgr.shape[:2]
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
    y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
    return frame_bgr[y1:y2, x1:x2]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPRegressor().to(device)
    model_path = HEALTH_ROOT / "hp_model.pth"
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()

    resolved_title = find_window_title(WINDOW_TITLE, EXE_NAME)
    print(f"[capture] backend=windows-capture window='{resolved_title}'")
    print("[capture] no resize/move will be applied to game window")

    camera = WindowsCaptureAdapter(window_name=resolved_title)
    time.sleep(0.3)

    print("Realtime HP test started. Press 'q' to quit.")
    print("Windows: [Game Frame] [HP Crop] [Model Input]")

    with torch.no_grad():
        while True:
            frame = camera.get_latest_bgr()
            if frame is None:
                time.sleep(0.005)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            hp_crop = crop_hp(frame, HP_RECT)
            if hp_crop.size == 0:
                continue

            img = cv2.resize(hp_crop, (160, 32), interpolation=cv2.INTER_AREA)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0

            pred_ratio = model(img_t.unsqueeze(0)).item()
            pred_ratio = float(np.clip(pred_ratio, 0.0, 1.0))

            hp_display = hp_crop.copy()
            cv2.putText(
                hp_display,
                f"HP: {pred_ratio:.2%}",
                (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Full game frame view: draw the configured HP ROI so you can verify location.
            game_display = frame.copy()
            x1, y1, x2, y2 = HP_RECT
            cv2.rectangle(game_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                game_display,
                f"HP ROI {HP_RECT}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                game_display,
                f"Pred: {pred_ratio:.2%}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Model input view: exactly what is fed into model (160x32 BGR).
            model_input_display = img.copy()
            cv2.putText(
                model_input_display,
                "Model Input 160x32",
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Game Frame", game_display)
            cv2.imshow("HP Crop", cv2.resize(hp_display, (max(400, hp_display.shape[1] * 3), max(80, hp_display.shape[0] * 3))))
            cv2.imshow("Model Input", cv2.resize(model_input_display, (640, 128), interpolation=cv2.INTER_NEAREST))
            print(f"\rpred_hp_ratio={pred_ratio:.4f}", end="")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
