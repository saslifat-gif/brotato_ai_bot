import argparse
import asyncio
import ctypes
import difflib
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Use a project-local PaddleX cache to avoid permission issues in user profile cache.
_LOCAL_PDX_CACHE = str((Path(__file__).resolve().parent / ".paddlex_cache").resolve())
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", _LOCAL_PDX_CACHE)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
try:
    os.makedirs(os.environ.get("PADDLE_PDX_CACHE_HOME", _LOCAL_PDX_CACHE), exist_ok=True)
except Exception:
    pass

PaddleOCR = None

try:
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter
except Exception:
    Language = None
    BitmapPixelFormat = None
    SoftwareBitmap = None
    OcrEngine = None
    DataWriter = None

try:
    import windows_capture as wc
except Exception:
    wc = None


SHOP_BUY_POINTS = [
    (195, 583),
    (560, 577),
    (912, 583),
    (1291, 574),
]
SHOP_REFRESH_RECT = (1324, 63, 1457, 104)
SHOP_GO_RECT = (1486, 821, 1884, 887)
SHOP_CARD_W = 360
SHOP_CARD_UP = 360
SHOP_CARD_DOWN = 80
SHOP_TOOLTIP_RECT = (26, 145, 1459, 687)
SHOP_ICON_HALF = 42
SHOP_ICON_HASH_HAMMING = 8


PRIMITIVE_WEAPON_TOKENS = [
    "原始",
    "primitive",
    "木棍",
    "棍",
    "stick",
    "长矛",
    "矛",
    "spear",
    "弹弓",
    "slingshot",
    "火把",
    "torch",
]

NON_PRIMITIVE_HINT_TOKENS = [
    "枪",
    "gun",
    "手枪",
    "冲锋",
    "shotgun",
    "smg",
    "laser",
    "法杖",
    "wand",
    "rocket",
]

MELEE_HINT_TOKENS = [
    "近战",
    "melee",
    "伤害",
    "damage",
]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


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


def resolve_hwnd(title: str, exe_name: str):
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if hwnd:
        return hwnd
    return find_hwnd_by_exe(exe_name)


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


def norm_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = (
        s.replace(" ", "")
        .replace("\n", "")
        .replace("：", ":")
        .replace("％", "%")
        .replace("，", ",")
        .replace("。", ".")
        .replace("傷害", "伤害")
        .replace("傷", "伤")
    )
    return s


def _best_substring_ratio(text: str, token: str) -> float:
    t = str(text or "")
    k = str(token or "")
    if not t or not k:
        return 0.0
    if k in t:
        return 1.0
    lt = len(t)
    lk = len(k)
    if lt == 0 or lk == 0:
        return 0.0
    # Slide around token-length windows to tolerate OCR character noise.
    wmin = max(1, lk - 1)
    wmax = min(lt, lk + 2)
    best = 0.0
    for w in range(wmin, wmax + 1):
        if w > lt:
            continue
        for i in range(0, lt - w + 1):
            sub = t[i : i + w]
            r = difflib.SequenceMatcher(None, sub, k).ratio()
            if r > best:
                best = r
    return float(best)


def _token_hit(text: str, token: str, fuzzy_ratio: float) -> bool:
    t = norm_text(text)
    k = norm_text(token)
    if not t or not k:
        return False
    if k in t:
        return True
    return _best_substring_ratio(t, k) >= float(fuzzy_ratio)


def extract_text_score_pairs(raw: Any) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []

    def push(t: Any, s: Any):
        try:
            txt = str(t).strip()
            conf = float(s)
        except Exception:
            return
        if txt:
            out.append((txt, float(np.clip(conf, 0.0, 1.0))))

    def visit(node: Any):
        if node is None:
            return
        if isinstance(node, str):
            push(node, 1.0)
            return
        if isinstance(node, dict):
            texts = None
            scores = None
            for tk in ("rec_texts", "texts", "text", "rec_text", "transcription", "label"):
                if tk in node:
                    texts = node.get(tk)
                    break
            for sk in ("rec_scores", "scores", "score", "rec_score", "confidence", "confidences"):
                if sk in node:
                    scores = node.get(sk)
                    break
            if texts is not None:
                if isinstance(texts, (list, tuple)):
                    if isinstance(scores, (list, tuple)):
                        for t, s in zip(texts, scores):
                            push(t, s)
                    else:
                        for t in texts:
                            push(t, 1.0 if scores is None else scores)
                else:
                    push(texts, 1.0 if scores is None else scores)
            for v in node.values():
                visit(v)
            return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)) and len(node[1]) >= 2:
                push(node[1][0], node[1][1])
            if len(node) == 2 and isinstance(node[0], str):
                push(node[0], node[1])
            for it in node:
                visit(it)
            return
        to_dict = getattr(node, "to_dict", None)
        if callable(to_dict):
            try:
                visit(to_dict())
                return
            except Exception:
                pass
        as_dict = getattr(node, "__dict__", None)
        if isinstance(as_dict, dict) and len(as_dict) > 0:
            visit(as_dict)
            return
        rec_texts = getattr(node, "rec_texts", None)
        rec_scores = getattr(node, "rec_scores", None)
        if rec_texts is not None:
            if isinstance(rec_texts, (list, tuple)):
                if isinstance(rec_scores, (list, tuple)):
                    for t, s in zip(rec_texts, rec_scores):
                        push(t, s)
                else:
                    for t in rec_texts:
                        push(t, 1.0 if rec_scores is None else rec_scores)
            else:
                push(rec_texts, 1.0 if rec_scores is None else rec_scores)
            return

    visit(raw)
    return out


def run_ocr(ocr: Any, img_bgr: np.ndarray):
    pred_fn = getattr(ocr, "predict", None)
    if callable(pred_fn):
        try:
            raw = pred_fn(img_bgr)
        except TypeError:
            raw = pred_fn([img_bgr])
        if hasattr(raw, "__iter__") and not isinstance(raw, (list, tuple, dict, str, bytes)):
            try:
                raw = list(raw)
            except Exception:
                pass
        return raw
    ocr_fn = getattr(ocr, "ocr", None)
    if callable(ocr_fn):
        return ocr_fn(img_bgr)
    return None


def slot_card_patch(
    frame_bgr: np.ndarray,
    pt: Tuple[int, int],
    card_w: int,
    card_up: int,
    card_down: int,
) -> np.ndarray:
    x, y = int(pt[0]), int(pt[1])
    h, w = frame_bgr.shape[:2]
    half_w = max(40, int(card_w) // 2)
    x1 = int(np.clip(x - half_w, 0, max(0, w - 1)))
    x2 = int(np.clip(x + half_w, x1 + 1, w))
    y1 = int(np.clip(y - max(40, int(card_up)), 0, max(0, h - 1)))
    y2 = int(np.clip(y + max(5, int(card_down)), y1 + 1, h))
    return frame_bgr[y1:y2, x1:x2]


def slot_icon_patch(frame_bgr: np.ndarray, pt: Tuple[int, int], icon_half: int) -> np.ndarray:
    x, y = int(pt[0]), int(pt[1])
    h, w = frame_bgr.shape[:2]
    ih = max(12, int(icon_half))
    x1 = int(np.clip(x - ih, 0, max(0, w - 1)))
    x2 = int(np.clip(x + ih, x1 + 1, w))
    y1 = int(np.clip(y - int(2.5 * ih), 0, max(0, h - 1)))
    y2 = int(np.clip(y - int(0.9 * ih), y1 + 1, h))
    return frame_bgr[y1:y2, x1:x2]


def dhash64(bgr: np.ndarray) -> int:
    if bgr is None or bgr.size == 0:
        return 0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    d = g[:, 1:] > g[:, :-1]
    bits = 0
    idx = 0
    for yy in range(8):
        for xx in range(8):
            if bool(d[yy, xx]):
                bits |= (1 << idx)
            idx += 1
    return int(bits)


def hamming64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def icon_star_color_name(icon_bgr: np.ndarray) -> str:
    if icon_bgr is None or icon_bgr.size == 0:
        return "none"
    h, w = icon_bgr.shape[:2]
    y1 = int(np.clip(0.70 * h, 0, h - 1))
    y2 = int(np.clip(0.98 * h, y1 + 1, h))
    x1 = int(np.clip(0.05 * w, 0, w - 1))
    x2 = int(np.clip(0.95 * w, x1 + 1, w))
    roi = icon_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        roi = icon_bgr
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    m = (sat > 60) & (val > 80)
    if int(np.count_nonzero(m)) < 8:
        return "gray"
    hue = hsv[:, :, 0][m]
    h_mean = float(np.mean(hue))
    if h_mean < 10 or h_mean >= 170:
        return "red"
    if h_mean < 22:
        return "orange"
    if h_mean < 35:
        return "yellow"
    if h_mean < 78:
        return "green"
    if h_mean < 105:
        return "cyan"
    if h_mean < 135:
        return "blue"
    if h_mean < 165:
        return "purple"
    return "gray"


def star_level_from_color(color_name: str) -> int:
    c = str(color_name or "").lower()
    if c in ("red", "orange", "yellow"):
        return 3
    if c in ("purple",):
        return 2
    if c in ("blue", "cyan"):
        return 1
    if c in ("green",):
        return 1
    return 0


def find_known_item_idx(known_items: List[Dict[str, Any]], icon_hash: int, ham_thr: int) -> int:
    for i, item in enumerate(known_items):
        h = int(item.get("hash", 0))
        if hamming64(icon_hash, h) <= int(ham_thr):
            return i
    return -1


def score_slot_for_primitive(pairs: List[Tuple[str, float]], fuzzy_ratio: float) -> Dict[str, float]:
    primitive = 0.0
    melee = 0.0
    non_primitive = 0.0
    text_conf = 0.0
    lines_norm: List[str] = []
    for txt, conf in pairs:
        t = norm_text(txt)
        c = float(np.clip(conf, 0.0, 1.0))
        lines_norm.append(t)
        text_conf = max(text_conf, c)
        if any(_token_hit(t, tok, fuzzy_ratio) for tok in PRIMITIVE_WEAPON_TOKENS):
            primitive = max(primitive, c)
        if any(_token_hit(t, tok, fuzzy_ratio) for tok in MELEE_HINT_TOKENS):
            melee = max(melee, c)
        if any(_token_hit(t, tok, fuzzy_ratio) for tok in NON_PRIMITIVE_HINT_TOKENS):
            non_primitive = max(non_primitive, c)

    # Strictly prefer primitive-class signals.
    total = 2.0 * primitive + 0.4 * melee - 1.2 * non_primitive
    return {
        "score": float(total),
        "primitive": float(primitive),
        "melee": float(melee),
        "non_primitive": float(non_primitive),
        "text_conf": float(text_conf),
        "has_primitive": 1.0 if primitive > 0.0 else 0.0,
        "line_count": float(len(lines_norm)),
    }


def click_client_rect(hwnd, rect: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = rect
    lx, rx = sorted((int(x1), int(x2)))
    ty, by = sorted((int(y1), int(y2)))
    cx = int(np.random.randint(lx, rx + 1))
    cy = int(np.random.randint(ty, by + 1))
    pt = POINT()
    pt.x = cx
    pt.y = cy
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    ctypes.windll.user32.SetCursorPos(int(pt.x), int(pt.y))
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.01)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up


def move_cursor_client_point(hwnd, pos: Tuple[int, int]):
    x, y = int(pos[0]), int(pos[1])
    pt = POINT()
    pt.x = x
    pt.y = y
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    ctypes.windll.user32.SetCursorPos(int(pt.x), int(pt.y))


def click_client_point(hwnd, pos: Tuple[int, int]):
    x, y = int(pos[0]), int(pos[1])
    pt = POINT()
    pt.x = x
    pt.y = y
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    ctypes.windll.user32.SetCursorPos(int(pt.x), int(pt.y))
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.01)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


def parse_rect(raw: str, default_rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    s = str(raw or "").strip()
    if not s:
        return default_rect
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            return default_rect
        x1, y1, x2, y2 = parts
        if x2 <= x1 or y2 <= y1:
            return default_rect
        return (x1, y1, x2, y2)
    except Exception:
        return default_rect


def parse_points(raw: str, default_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    s = str(raw or "").strip()
    if not s:
        return list(default_points)
    out: List[Tuple[int, int]] = []
    try:
        chunks = [c.strip() for c in s.split(";") if c.strip()]
        for ch in chunks:
            parts = [int(x.strip()) for x in ch.split(",")]
            if len(parts) != 2:
                continue
            out.append((int(parts[0]), int(parts[1])))
    except Exception:
        return list(default_points)
    return out if len(out) > 0 else list(default_points)


def crop_rect(frame_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = rect
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    return frame_bgr[y1:y2, x1:x2]


def ocr_pairs_from_patch(
    ocr: Any,
    patch_bgr: np.ndarray,
    min_conf: float,
    error_bucket: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    if patch_bgr is None or patch_bgr.size == 0:
        return []
    try:
        img = cv2.resize(patch_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        th_inv = cv2.bitwise_not(th)
        candidates = [
            img,
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(th_inv, cv2.COLOR_GRAY2BGR),
        ]
    except Exception as e:
        if error_bucket is not None:
            error_bucket.append(f"preprocess:{e}")
        candidates = [patch_bgr]

    best_pairs: List[Tuple[str, float]] = []
    best_key = (-1, -1.0)
    for cand in candidates:
        try:
            raw = run_ocr(ocr, cand)
            ext = extract_text_score_pairs(raw)
            pairs = [(txt, conf) for (txt, conf) in ext if float(conf) >= float(min_conf)]
            key = (len(pairs), max((float(c) for _, c in pairs), default=0.0))
            if key > best_key:
                best_key = key
                best_pairs = pairs
        except Exception as e:
            if error_bucket is not None:
                msg = str(e).replace("\n", " ").strip()
                if msg:
                    error_bucket.append(msg)
            continue
    return best_pairs


class WinMediaOCR:
    def __init__(self, lang: str):
        if OcrEngine is None or Language is None or SoftwareBitmap is None or DataWriter is None:
            raise RuntimeError(
                "Windows.Media.Ocr dependencies are missing. Install: "
                "pip install winrt-runtime winrt-Windows.Media.Ocr "
                "winrt-Windows.Graphics.Imaging winrt-Windows.Storage.Streams "
                "winrt-Windows.Globalization winrt-Windows.Foundation "
                "winrt-Windows.Foundation.Collections"
            )
        self.lang = str(lang or "zh-Hans")
        self.engine = OcrEngine.try_create_from_language(Language(self.lang))
        if self.engine is None:
            self.engine = OcrEngine.try_create_from_user_profile_languages()
        if self.engine is None:
            raise RuntimeError(f"Cannot create Windows.Media.Ocr engine for lang='{self.lang}'")
        self._loop = asyncio.new_event_loop()

    async def _recognize_async(self, sb: Any):
        return await self.engine.recognize_async(sb)

    def _to_software_bitmap(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            raise RuntimeError("empty image")
        if len(img_bgr.shape) == 2:
            bgra = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGRA)
        elif img_bgr.shape[2] == 4:
            bgra = img_bgr
        else:
            bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        h, w = bgra.shape[:2]
        writer = DataWriter()
        writer.write_bytes(bgra.tobytes())
        buf = writer.detach_buffer()
        return SoftwareBitmap.create_copy_from_buffer(buf, BitmapPixelFormat.BGRA8, int(w), int(h))

    def predict(self, img_bgr: np.ndarray):
        sb = self._to_software_bitmap(img_bgr)
        res = self._loop.run_until_complete(self._recognize_async(sb))
        texts: List[str] = []
        scores: List[float] = []
        try:
            lines = list(res.lines)
        except Exception:
            lines = []
        for ln in lines:
            txt = str(getattr(ln, "text", "")).strip()
            if txt:
                texts.append(txt)
                scores.append(1.0)
        if len(texts) == 0:
            txt = str(getattr(res, "text", "")).strip()
            if txt:
                for part in txt.splitlines():
                    p = str(part).strip()
                    if p:
                        texts.append(p)
                        scores.append(1.0)
        return {"rec_texts": texts, "rec_scores": scores}

    def close(self):
        try:
            self._loop.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Shop policy test: primitive weapon buy checker.")
    p.add_argument("--window-title", default="Brotato")
    p.add_argument("--exe-name", default="Brotato.exe")
    p.add_argument("--video", default="", help="Optional video path; if set, use video instead of window capture.")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--ocr-backend", choices=["winmedia", "paddle"], default="winmedia")
    p.add_argument("--ocr-lang", default="zh-Hans", help="winmedia: e.g. zh-Hans/en-US; paddle: ch/en/...")
    p.add_argument("--ocr-device", default="cpu", help="cpu or gpu:0")
    p.add_argument("--min-conf", type=float, default=0.20)
    p.add_argument("--primitive-min", type=float, default=0.45, help="Need primitive confidence >= this to buy.")
    p.add_argument("--buy-min-score", type=float, default=0.35)
    p.add_argument("--fuzzy-ratio", type=float, default=0.72, help="OCR fuzzy match threshold for token hit.")
    p.add_argument("--refresh-max", type=int, default=2)
    p.add_argument("--max-buys", type=int, default=4)
    p.add_argument("--cooldown", type=float, default=0.8)
    p.add_argument(
        "--buy-points",
        default="195,583;560,577;912,583;1291,574",
        help="Semicolon-separated slot center points: x1,y1;x2,y2;...",
    )
    p.add_argument("--slot-offset-x", type=int, default=0, help="Shift all slot click points by x.")
    p.add_argument("--slot-offset-y", type=int, default=0, help="Shift all slot click points by y.")
    p.add_argument("--ocr-point-dx", type=int, default=0, help="OCR sample point x offset relative to click point.")
    p.add_argument("--ocr-point-dy", type=int, default=-120, help="OCR sample point y offset relative to click point.")
    p.add_argument("--card-w", type=int, default=SHOP_CARD_W, help="OCR crop width around OCR sample point.")
    p.add_argument("--card-up", type=int, default=SHOP_CARD_UP, help="OCR crop extends this many px upward.")
    p.add_argument("--card-down", type=int, default=SHOP_CARD_DOWN, help="OCR crop extends this many px downward.")
    p.add_argument("--icon-half", type=int, default=SHOP_ICON_HALF, help="Half size for item icon crop.")
    p.add_argument("--hash-hamming", type=int, default=SHOP_ICON_HASH_HAMMING, help="Hash distance threshold for duplicate item.")
    p.add_argument("--dup-weight", type=float, default=0.8, help="Extra score for duplicate/upgrade candidate.")
    p.add_argument("--star-weight", type=float, default=0.25, help="Extra score for higher star color level.")
    p.add_argument("--hover-scan", action=argparse.BooleanOptionalAction, default=False, help="Hover each slot and OCR tooltip panel.")
    p.add_argument("--hover-wait", type=float, default=0.12, help="Seconds to wait after moving cursor to slot.")
    p.add_argument("--force-hover", action=argparse.BooleanOptionalAction, default=False, help="Allow hover scan even when click is off.")
    p.add_argument(
        "--tooltip-rect",
        default="26,145,1459,687",
        help="Tooltip OCR rect in client coords: x1,y1,x2,y2",
    )
    p.add_argument("--refresh-rect", default="1324,63,1457,104", help="Refresh rect x1,y1,x2,y2")
    p.add_argument("--go-rect", default="1486,821,1884,887", help="Go rect x1,y1,x2,y2")
    p.add_argument("--debug-ocr", action=argparse.BooleanOptionalAction, default=False, help="Print OCR lines for each slot.")
    p.add_argument("--debug-hover", action=argparse.BooleanOptionalAction, default=False, help="Print hover/cursor errors.")
    p.add_argument("--click", action=argparse.BooleanOptionalAction, default=False, help="Enable real clicks.")
    return p.parse_args()


def _map_lang_for_paddle(lang: str) -> str:
    s = str(lang or "").strip().lower()
    if s in ("zh", "zh-cn", "zh_hans", "zh-hans", "ch", "cn"):
        return "ch"
    if s in ("en-us", "en_us", "en"):
        return "en"
    return str(lang or "ch")


def init_ocr(backend: str, lang: str, device: str):
    global PaddleOCR
    b = str(backend or "winmedia").strip().lower()
    if b == "winmedia":
        ocr = WinMediaOCR(lang=lang)
        print(f"[ocr] backend=winmedia lang={lang}")
        return ocr

    if PaddleOCR is None:
        try:
            from paddleocr import PaddleOCR as _PaddleOCR

            PaddleOCR = _PaddleOCR
        except Exception as e:
            raise RuntimeError(f"paddleocr is not installed: {e}")

    plang = _map_lang_for_paddle(lang)
    attempts = [
        {
            "lang": plang,
            "use_textline_orientation": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "device": device,
        },
        {"lang": plang, "use_angle_cls": False, "show_log": False, "use_gpu": device.startswith("gpu")},
        {"lang": plang},
    ]
    last_error = ""
    for kwargs in attempts:
        try:
            ocr = PaddleOCR(**kwargs)
            print(f"[ocr] backend=paddle lang={plang} args={list(kwargs.keys())}")
            return ocr
        except Exception as e:
            last_error = str(e)
    raise RuntimeError(f"OCR init failed: {last_error}")


def build_panel(
    frame_bgr: np.ndarray,
    slot_infos: List[Dict[str, Any]],
    best_idx: int,
    action: str,
    counters: Dict[str, int],
    fps: float,
    clicking: bool,
    slot_click_points: List[Tuple[int, int]],
    slot_ocr_points: List[Tuple[int, int]],
):
    vis = frame_bgr.copy()
    for i, pt in enumerate(slot_click_points):
        x, y = int(pt[0]), int(pt[1])
        color = (0, 220, 0) if i == best_idx else (180, 180, 180)
        cv2.circle(vis, (x, y), 9, color, 2)
        cv2.putText(vis, f"S{i+1}", (x - 18, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for i, pt in enumerate(slot_ocr_points):
        x, y = int(pt[0]), int(pt[1])
        color = (0, 180, 255) if i == best_idx else (120, 120, 120)
        cv2.drawMarker(vis, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=11, thickness=1)

    h, w = vis.shape[:2]
    panel_h = min(h, 280)
    cv2.rectangle(vis, (0, 0), (min(w - 1, 980), panel_h), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"Primitive Shop Test | fps={fps:.1f} | click={'on' if clicking else 'off'} | action={action}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        (
            f"decisions={counters['decisions']} buy={counters['buy']} "
            f"refresh={counters['refresh']} go={counters['go']} dup_buy={counters.get('dup_buy', 0)} "
            f"hover_fail={counters.get('hover_fail', 0)} ocr_fail={counters.get('ocr_fail', 0)}"
        ),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (180, 255, 180),
        1,
    )
    cv2.putText(
        vis,
        "circle=click point, cross=OCR point",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 220, 255),
        1,
    )
    y = 96
    for info in slot_infos:
        sid = int(info.get("slot", 0))
        color = (0, 220, 0) if (sid - 1) == best_idx else (220, 220, 220)
        cv2.putText(
            vis,
            (
                f"S{sid}: score={info['score']:.3f} prim={info['primitive']:.2f} "
                f"melee={info['melee']:.2f} non_prim={info['non_primitive']:.2f} "
                f"dup={info.get('duplicate', 0.0):.0f} star={info.get('star_level', 0)} "
                f"txt={info.get('text_conf', 0.0):.2f} n={int(info.get('line_count', 0))} "
                f"src={info.get('scan_src', '-')}"
            ),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            color,
            1,
        )
        y += 24
        if y > panel_h - 10:
            break
    return vis


def decide_action(
    slot_infos: List[Dict[str, Any]],
    buy_count: int,
    refresh_count: int,
    primitive_min: float,
    buy_min_score: float,
    refresh_max: int,
    max_buys: int,
):
    best_idx = -1
    best_score = -1e9
    best = None
    for i, info in enumerate(slot_infos):
        if float(info["score"]) > best_score:
            best_score = float(info["score"])
            best_idx = i
            best = info

    if best is None:
        return "wait", -1

    can_buy = (
        float(best["primitive"]) >= float(primitive_min)
        and float(best["score"]) >= float(buy_min_score)
        and buy_count < int(max_buys)
    )
    if can_buy:
        return "buy", best_idx
    if refresh_count < int(refresh_max) and buy_count < 1:
        return "refresh", -1
    return "go", -1


def main():
    args = parse_args()
    if (not bool(args.click)) and bool(args.hover_scan) and (not bool(args.force_hover)):
        args.hover_scan = False
        print("[safety] click=off -> hover_scan forced off (use --force-hover to override).")

    ocr = init_ocr(args.ocr_backend, args.ocr_lang, args.ocr_device)
    base_buy_points = parse_points(args.buy_points, SHOP_BUY_POINTS)
    slot_click_points = [
        (int(x + args.slot_offset_x), int(y + args.slot_offset_y))
        for (x, y) in base_buy_points
    ]
    slot_ocr_points = [
        (int(x + args.slot_offset_x + args.ocr_point_dx), int(y + args.slot_offset_y + args.ocr_point_dy))
        for (x, y) in base_buy_points
    ]
    print(
        f"[config] ocr_backend={args.ocr_backend} ocr_lang={args.ocr_lang} "
        f"sample_every={args.sample_every} min_conf={args.min_conf:.2f} "
        f"primitive_min={args.primitive_min:.2f} buy_min_score={args.buy_min_score:.2f} "
        f"fuzzy_ratio={args.fuzzy_ratio:.2f} refresh_max={args.refresh_max} "
        f"max_buys={args.max_buys} click={args.click}"
    )
    tooltip_rect = parse_rect(args.tooltip_rect, SHOP_TOOLTIP_RECT)
    refresh_rect = parse_rect(args.refresh_rect, SHOP_REFRESH_RECT)
    go_rect = parse_rect(args.go_rect, SHOP_GO_RECT)
    print(
        f"[config] slot_offset=({args.slot_offset_x},{args.slot_offset_y}) "
        f"ocr_point_delta=({args.ocr_point_dx},{args.ocr_point_dy}) "
        f"crop(w={args.card_w},up={args.card_up},down={args.card_down}) "
        f"hover_scan={args.hover_scan} force_hover={args.force_hover} "
        f"tooltip_rect={tooltip_rect} slots={len(slot_click_points)}"
    )
    print(f"[config] refresh_rect={refresh_rect} go_rect={go_rect}")
    print(
        f"[config] icon_half={args.icon_half} hash_hamming={args.hash_hamming} "
        f"dup_weight={args.dup_weight:.2f} star_weight={args.star_weight:.2f}"
    )
    print(f"[config] pdx_cache_home={os.environ.get('PADDLE_PDX_CACHE_HOME', '')}")

    cap = None
    cam = None
    hwnd = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        print(f"[source] video={args.video}")
    else:
        title = resolve_window_title(args.window_title, args.exe_name)
        hwnd = resolve_hwnd(title, args.exe_name)
        cam = WindowsCaptureAdapter(title)
        print(f"[source] window='{title}' backend=windows-capture hwnd={'ok' if hwnd else 'none'}")
        if args.hover_scan and hwnd is None:
            print("[warn] hover_scan enabled but hwnd is None, fallback to card OCR only.")
        time.sleep(0.2)

    frame_idx = 0
    infer_idx = 0
    t_prev = time.perf_counter()
    last_action_ts = 0.0
    buy_count = 0
    refresh_count = 0
    counters = {"decisions": 0, "buy": 0, "refresh": 0, "go": 0, "dup_buy": 0, "hover_fail": 0, "ocr_fail": 0}
    known_items: List[Dict[str, Any]] = []
    auto_click = bool(args.click)
    last_action = "idle"
    last_panel = None
    hover_last_err = ""
    stop_now = {"flag": False}

    def _on_sigint(_signum, _frame):
        stop_now["flag"] = True
        print("[control] Ctrl+C received, stopping...")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        pass

    try:
        while True:
            if stop_now["flag"]:
                break
            if cap is not None:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
            else:
                frame_bgr = cam.get_latest_bgr()
                if frame_bgr is None:
                    time.sleep(0.005)
                    if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

            frame_idx += 1
            slot_infos: List[Dict[str, Any]] = []
            best_idx = -1

            if frame_idx % max(1, args.sample_every) == 0:
                infer_idx += 1
                for i, pt in enumerate(slot_ocr_points):
                    icon = slot_icon_patch(frame_bgr, slot_click_points[i], args.icon_half)
                    icon_hash = dhash64(icon)
                    known_idx = find_known_item_idx(known_items, icon_hash, args.hash_hamming)
                    duplicate = 1.0 if known_idx >= 0 else 0.0
                    star_color = icon_star_color_name(icon)
                    star_level = star_level_from_color(star_color)

                    card = slot_card_patch(
                        frame_bgr=frame_bgr,
                        pt=pt,
                        card_w=args.card_w,
                        card_up=args.card_up,
                        card_down=args.card_down,
                    )
                    if card is None or card.size == 0:
                        info = {
                            "slot": i + 1,
                            "score": -1.0,
                            "primitive": 0.0,
                            "melee": 0.0,
                            "non_primitive": 0.0,
                            "duplicate": duplicate,
                            "star_level": star_level,
                            "star_color": star_color,
                            "icon_hash": int(icon_hash),
                        }
                        slot_infos.append(info)
                        continue

                    scan_sources: List[Tuple[str, np.ndarray]] = [("card", card)]
                    scan_src = "card"
                    if args.hover_scan and (hwnd is not None) and (cam is not None):
                        try:
                            move_cursor_client_point(hwnd, slot_click_points[i])
                            time.sleep(max(0.0, float(args.hover_wait)))
                            f2 = cam.get_latest_bgr()
                            if f2 is not None and f2.size > 0:
                                tip = crop_rect(f2, tooltip_rect)
                                if tip is not None and tip.size > 0:
                                    scan_sources.insert(0, ("tooltip", tip))
                            else:
                                counters["hover_fail"] = int(counters.get("hover_fail", 0)) + 1
                                hover_last_err = "empty_frame_after_hover"
                        except Exception as e:
                            counters["hover_fail"] = int(counters.get("hover_fail", 0)) + 1
                            hover_last_err = str(e)
                            if args.debug_hover and counters["hover_fail"] <= 20:
                                print(f"[hover] fail slot={i+1}: {hover_last_err}")

                    best_pairs: List[Tuple[str, float]] = []
                    best_src = "card"
                    best_key = (-1.0, -1.0, -1.0, -1.0)
                    slot_errs: List[str] = []
                    for src_name, src_patch in scan_sources:
                        src_pairs = ocr_pairs_from_patch(
                            ocr=ocr,
                            patch_bgr=src_patch,
                            min_conf=float(args.min_conf),
                            error_bucket=slot_errs,
                        )
                        src_sc = score_slot_for_primitive(src_pairs, args.fuzzy_ratio)
                        src_key = (
                            float(src_sc.get("primitive", 0.0)),
                            float(src_sc.get("score", 0.0)),
                            float(src_sc.get("text_conf", 0.0)),
                            float(src_sc.get("line_count", 0.0)),
                        )
                        if src_key > best_key:
                            best_key = src_key
                            best_src = src_name
                            best_pairs = src_pairs

                    pairs = best_pairs
                    scan_src = best_src
                    if len(pairs) == 0:
                        counters["ocr_fail"] = int(counters.get("ocr_fail", 0)) + 1

                    sc = score_slot_for_primitive(pairs, args.fuzzy_ratio)
                    sc["duplicate"] = float(duplicate)
                    sc["star_level"] = int(star_level)
                    sc["star_color"] = str(star_color)
                    sc["icon_hash"] = int(icon_hash)
                    sc["known_idx"] = int(known_idx)
                    sc["score"] = float(sc["score"]) + float(args.dup_weight) * float(duplicate) + float(args.star_weight) * float(star_level)
                    top_lines = [norm_text(t) for (t, _) in sorted(pairs, key=lambda x: -float(x[1]))[:3]]
                    info = {"slot": i + 1}
                    info.update(sc)
                    info["top_lines"] = top_lines
                    info["scan_src"] = scan_src
                    info["ocr_err"] = " | ".join(slot_errs[:2]) if slot_errs else ""
                    slot_infos.append(info)

                action, best_idx = decide_action(
                    slot_infos=slot_infos,
                    buy_count=buy_count,
                    refresh_count=refresh_count,
                    primitive_min=args.primitive_min,
                    buy_min_score=args.buy_min_score,
                    refresh_max=args.refresh_max,
                    max_buys=args.max_buys,
                )

                now = time.time()
                can_act = (now - last_action_ts) >= float(args.cooldown)
                if action != "wait":
                    counters["decisions"] += 1

                if action == "buy" and best_idx >= 0 and can_act:
                    last_action = f"buy_{best_idx + 1}"
                    counters["buy"] += 1
                    buy_count += 1
                    last_action_ts = now
                    best_info = slot_infos[best_idx] if (0 <= best_idx < len(slot_infos)) else None
                    if best_info is not None:
                        bh = int(best_info.get("icon_hash", 0))
                        b_star = int(best_info.get("star_level", 0))
                        b_color = str(best_info.get("star_color", "none"))
                        kidx = find_known_item_idx(known_items, bh, args.hash_hamming)
                        if kidx >= 0:
                            known_items[kidx]["count"] = int(known_items[kidx].get("count", 1)) + 1
                            if b_star > int(known_items[kidx].get("star_level", 0)):
                                known_items[kidx]["star_level"] = b_star
                                known_items[kidx]["star_color"] = b_color
                            counters["dup_buy"] = int(counters.get("dup_buy", 0)) + 1
                        else:
                            known_items.append(
                                {
                                    "hash": int(bh),
                                    "count": 1,
                                    "star_level": int(b_star),
                                    "star_color": b_color,
                                }
                            )
                    if auto_click and hwnd:
                        click_client_point(hwnd, slot_click_points[best_idx])
                elif action == "refresh" and can_act:
                    last_action = "refresh"
                    counters["refresh"] += 1
                    refresh_count += 1
                    last_action_ts = now
                    if auto_click and hwnd:
                        click_client_rect(hwnd, refresh_rect)
                elif action == "go" and can_act:
                    last_action = "go"
                    counters["go"] += 1
                    buy_count = 0
                    refresh_count = 0
                    last_action_ts = now
                    if auto_click and hwnd:
                        click_client_rect(hwnd, go_rect)

                if args.print_every > 0 and infer_idx % args.print_every == 0:
                    best = slot_infos[best_idx] if (0 <= best_idx < len(slot_infos)) else None
                    if best is not None:
                        print(
                            f"[decision] {last_action} | best=S{best_idx+1} score={best['score']:.3f} "
                            f"prim={best['primitive']:.2f} melee={best['melee']:.2f} "
                            f"non_prim={best['non_primitive']:.2f} dup={best.get('duplicate', 0.0):.0f} "
                            f"star={best.get('star_level', 0)}({best.get('star_color', '-')}) "
                            f"txt={best.get('text_conf', 0.0):.2f} n={int(best.get('line_count', 0))} "
                            f"known={len(known_items)} hover_fail={counters.get('hover_fail', 0)} "
                            f"ocr_fail={counters.get('ocr_fail', 0)}"
                        )
                        if args.debug_hover and hover_last_err:
                            print(f"[hover] last_err={hover_last_err[:120]}")
                    if args.debug_ocr:
                        for info in slot_infos:
                            lines = info.get("top_lines", [])
                            line_show = " | ".join(lines[:2]) if lines else "<no-text>"
                            print(
                                f"  S{int(info['slot'])} txt={info.get('text_conf', 0.0):.2f} "
                                f"n={int(info.get('line_count', 0))} src={info.get('scan_src', '-')} "
                                f"dup={info.get('duplicate', 0.0):.0f} star={info.get('star_level', 0)} "
                                f"lines={line_show}"
                            )
                            if info.get("ocr_err"):
                                print(f"    err={str(info.get('ocr_err'))[:180]}")

            t_now = time.perf_counter()
            fps = 1.0 / max(1e-6, t_now - t_prev)
            t_prev = t_now
            if args.show:
                if len(slot_infos) == 0 and last_panel is not None:
                    vis = last_panel.copy()
                else:
                    vis = build_panel(
                        frame_bgr=frame_bgr,
                        slot_infos=slot_infos,
                        best_idx=best_idx,
                        action=last_action,
                        counters=counters,
                        fps=fps,
                        clicking=auto_click,
                        slot_click_points=slot_click_points,
                        slot_ocr_points=slot_ocr_points,
                    )
                    last_panel = vis.copy()
                cv2.imshow("Shop Primitive Policy Test", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    auto_click = not auto_click
                    print(f"[control] click={'on' if auto_click else 'off'}")
    finally:
        if cap is not None:
            cap.release()
        if cam is not None:
            cam.stop()
        try:
            close_fn = getattr(ocr, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
