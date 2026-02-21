import argparse
import ctypes
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import windows_capture as wc
except Exception:
    wc = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEALTH_ROOT = PROJECT_ROOT / "health"


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
        raise RuntimeError("Found hwnd by exe but title is empty")
    return out


class WindowsCaptureAdapter:
    def __init__(self, window_name: str):
        if wc is None:
            raise RuntimeError("windows-capture not installed. pip install windows-capture")
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


def topk_from_result(res, k: int = 3):
    probs = getattr(res, "probs", None)
    if probs is None:
        return []
    arr = probs.data.detach().float().cpu().numpy()
    idx = np.argsort(-arr)
    names = getattr(res, "names", {}) or {}
    out = []
    for i in idx[: max(1, int(k))]:
        label = names.get(int(i), str(int(i)))
        out.append((label, float(arr[int(i)])))
    return out


def draw_overlay(frame_bgr: np.ndarray, topk: list[tuple[str, float]], fps: float):
    h, w = frame_bgr.shape[:2]
    panel_h = 26 + 24 * max(2, len(topk))
    cv2.rectangle(frame_bgr, (0, 0), (min(w - 1, 520), min(h - 1, panel_h)), (0, 0, 0), -1)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y = 44
    for rank, (name, score) in enumerate(topk, 1):
        color = (0, 255, 0) if rank == 1 else (0, 255, 255)
        cv2.putText(
            frame_bgr,
            f"Top{rank}: {name} ({score:.3f})",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        y += 24
    return frame_bgr


def parse_args():
    p = argparse.ArgumentParser(description="Realtime YOLO classifier tester (window/video).")
    p.add_argument("--model", default="runs/classify/train/weights/best.pt")
    p.add_argument("--window-title", default="Brotato")
    p.add_argument("--exe-name", default="Brotato.exe")
    p.add_argument("--video", default="", help="Optional video file path. If set, use video instead of window capture.")
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--device", default="0", help="e.g. 0, cpu")
    p.add_argument("--sample-every", type=int, default=1, help="Infer every N frames")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-every", type=int, default=5, help="Console print interval (frames)")
    return p.parse_args()


def run_video(model: YOLO, args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    print(f"[source] video={args.video}")
    infer_idx = 0
    frame_idx = 0
    topk = []
    t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % max(1, args.sample_every) == 0:
            infer_idx += 1
            res = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            topk = topk_from_result(res, args.topk)
            if args.print_every > 0 and infer_idx % args.print_every == 0 and topk:
                print(" | ".join([f"{n}:{s:.3f}" for n, s in topk]))

        t_now = time.perf_counter()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        if args.show:
            vis = draw_overlay(frame.copy(), topk, fps)
            cv2.imshow("YOLO Classify Debug", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_window(model: YOLO, args):
    title = resolve_window_title(args.window_title, args.exe_name)
    print(f"[source] window='{title}' backend=windows-capture")
    cam = WindowsCaptureAdapter(title)
    time.sleep(0.2)

    infer_idx = 0
    frame_idx = 0
    topk = []
    t_prev = time.perf_counter()
    while True:
        frame = cam.get_latest_bgr()
        if frame is None:
            time.sleep(0.005)
            if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        frame_idx += 1
        if frame_idx % max(1, args.sample_every) == 0:
            infer_idx += 1
            res = model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            topk = topk_from_result(res, args.topk)
            if args.print_every > 0 and infer_idx % args.print_every == 0 and topk:
                print(" | ".join([f"{n}:{s:.3f}" for n, s in topk]))

        t_now = time.perf_counter()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        if args.show:
            vis = draw_overlay(frame.copy(), topk, fps)
            cv2.imshow("YOLO Classify Debug", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.stop()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        cand_health = (HEALTH_ROOT / model_path).resolve()
        model_path = cand_health if cand_health.exists() else (Path(__file__).resolve().parent / model_path).resolve()
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    print(f"[model] {model_path}")
    model = YOLO(str(model_path))

    if args.video:
        run_video(model, args)
    else:
        run_window(model, args)


if __name__ == "__main__":
    main()
