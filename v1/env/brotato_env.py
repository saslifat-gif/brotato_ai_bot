import ctypes
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.runtime_config import RuntimeConfig
from reward.reward_engine import RewardEngine
from runtime.debug_windows import (
    DEBUG_WINDOW_HP,
    DEBUG_WINDOW_RED_GREEN,
    DEBUG_WINDOW_SHOP,
    DEBUG_WINDOW_STATE,
    DebugWindowManager,
)
from runtime.input_driver import InputDriver, RECT
from runtime.stop_manager import StopManager
from shop.ocr_winmedia import ShopOcrWorker, WinMediaOCR, score_upgrade_text, normalize_text
from shop.shop_policy import ShopPolicy

try:
    import mss
except Exception:
    mss = None

try:
    import windows_capture as wc
except Exception:
    wc = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from env.roboflow_detector import RoboflowMonsterDetector
except Exception:
    RoboflowMonsterDetector = None

CONTROL_PANEL_WINDOW = "Control Panel"


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def list_monitor_rects() -> List[RECT]:
    monitors: List[RECT] = []

    def enum_cb(_hmonitor, _hdc, lprc, _data):
        monitors.append(lprc.contents)
        return True

    enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_long,
    )
    ctypes.windll.user32.EnumDisplayMonitors(None, None, enum_proc(enum_cb), 0)
    return monitors


def monitor_for_point(x: int, y: int) -> Tuple[int, Tuple[int, int]]:
    monitors = list_monitor_rects()
    for i, m in enumerate(monitors):
        if m.left <= x < m.right and m.top <= y < m.bottom:
            return i, (int(m.left), int(m.top))
    return 0, (0, 0)


def find_hwnd_by_exe(exe_name: str) -> Optional[int]:
    hwnd_found = None
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_long)

    def enum_cb(hwnd, _lparam):
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

    ctypes.windll.user32.EnumWindows(enum_proc(enum_cb), 0)
    return hwnd_found


def find_window_fuzzy(keyword: str) -> Optional[int]:
    hwnd_found = None
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_long)

    def enum_cb(hwnd, _lparam):
        nonlocal hwnd_found
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                if keyword.lower() in buff.value.lower():
                    hwnd_found = hwnd
                    return False
        return True

    ctypes.windll.user32.EnumWindows(enum_proc(enum_cb), 0)
    return hwnd_found


def force_game_window(title: str, exe_name: str, resize: bool, w: int, h: int) -> Optional[int]:
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if not hwnd:
        hwnd = find_window_fuzzy(title)
    if not hwnd and exe_name:
        hwnd = find_hwnd_by_exe(exe_name)
    if not hwnd:
        return None
    ctypes.windll.user32.ShowWindow(hwnd, 5)
    if resize:
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, int(w), int(h), 0x0002 | 0x0040)
    return int(hwnd)


def hwnd_client_screen_rect(hwnd: int) -> Tuple[int, int, int, int]:
    crect = RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(crect))
    tl = POINT()
    br = POINT()
    tl.x, tl.y = int(crect.left), int(crect.top)
    br.x, br.y = int(crect.right), int(crect.bottom)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(tl))
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(br))
    return (int(tl.x), int(tl.y), int(br.x), int(br.y))


class MSSCameraAdapter:
    def __init__(self, region: Tuple[int, int, int, int], target_fps: int = 60):
        if mss is None:
            raise RuntimeError("mss not installed")
        self.region = region
        self.target_fps = max(1, int(target_fps))
        self.interval = 1.0 / float(self.target_fps)
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        with mss.mss() as sct:
            l, t, r, b = self.region
            monitor = {
                "left": int(l),
                "top": int(t),
                "width": int(max(1, r - l)),
                "height": int(max(1, b - t)),
            }
            while self._running:
                t0 = time.perf_counter()
                raw = np.asarray(sct.grab(monitor))
                frame = raw[:, :, [2, 1, 0]]
                with self._lock:
                    self._latest = frame
                dt = time.perf_counter() - t0
                sleep_t = self.interval - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class WindowsCaptureAdapter:
    def __init__(self, monitor_idx: int, region_in_monitor: Tuple[int, int, int, int]):
        if wc is None:
            raise RuntimeError("windows-capture not installed")
        self._lock = threading.Lock()
        self._latest = None
        self._finished = False
        self._control = None
        self._region = region_in_monitor

        self._cap = wc.WindowsCapture(monitor_index=int(monitor_idx) + 1)

        @self._cap.event
        def on_frame_arrived(frame, control):
            if self._control is None:
                self._control = control
            fb = frame.convert_to_bgr().frame_buffer
            l, t, r, b = self._region
            h, w = fb.shape[:2]
            x1 = int(np.clip(l, 0, max(0, w - 1)))
            y1 = int(np.clip(t, 0, max(0, h - 1)))
            x2 = int(np.clip(r, x1 + 1, max(x1 + 1, w)))
            y2 = int(np.clip(b, y1 + 1, max(y1 + 1, h)))
            out = fb[y1:y2, x1:x2]
            rgb = np.ascontiguousarray(out[:, :, ::-1])
            with self._lock:
                self._latest = rgb

        @self._cap.event
        def on_closed():
            self._finished = True

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


class YOLOStateInfer:
    def __init__(self, model, image_size: int = 256, device: str = "0"):
        self.model = model
        self.image_size = int(max(64, image_size))
        self.device = str(device)

    @staticmethod
    def load(model_path: str, image_size: int = 256, device: str = "0"):
        if YOLO is None:
            raise RuntimeError("ultralytics is unavailable")
        p = Path(model_path)
        if not p.exists():
            raise RuntimeError(f"state model not found: {p}")
        model = YOLO(str(p))
        return YOLOStateInfer(model, image_size=image_size, device=device)

    def predict_top2(self, frame_bgr: np.ndarray) -> Tuple[str, float, str, float]:
        res = self.model.predict(frame_bgr, imgsz=self.image_size, device=self.device, verbose=False)[0]
        probs = getattr(res, "probs", None)
        if probs is None:
            return ("unknown", 0.0, "unknown", 0.0)
        arr = probs.data.detach().float().cpu().numpy()
        if arr.ndim != 1 or arr.shape[0] == 0:
            return ("unknown", 0.0, "unknown", 0.0)
        idx = np.argsort(-arr)
        i1 = int(idx[0])
        i2 = int(idx[1]) if arr.shape[0] > 1 else i1
        names = getattr(res, "names", {}) or {}
        n1 = str(names.get(i1, i1)).lower()
        n2 = str(names.get(i2, i2)).lower()
        return n1, float(arr[i1]), n2, float(arr[i2])


class BrotatoEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, cfg: RuntimeConfig, stop_manager: StopManager):
        super().__init__()
        self.cfg = cfg
        self.stop_manager = stop_manager

        self.obs_size = int(cfg.obs_size)
        self.obs_channels = int(cfg.obs_channels)
        self.obs_danger_enable = bool(cfg.obs_danger_enable)
        self.obs_mask_mode = bool(cfg.obs_mask_mode)
        self.obs_stack = int(cfg.obs_stack)
        self.action_space = spaces.Discrete(5)  # none,w,s,a,d
        obs_depth = self.obs_channels * self.obs_stack
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_size, self.obs_size, obs_depth),
            dtype=np.uint8,
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._proximity_weight: Optional[np.ndarray] = None  # cached radial weight map
        self._last_player_pos: Optional[Tuple[int, int]] = None  # last known player position for mask mode

        self.hwnd = force_game_window(
            title=cfg.window_title,
            exe_name=cfg.exe_name,
            resize=cfg.force_resize,
            w=cfg.window_w,
            h=cfg.window_h,
        )
        if not self.hwnd:
            raise RuntimeError(f"game window not found title={cfg.window_title} exe={cfg.exe_name}")

        self.game_region = hwnd_client_screen_rect(self.hwnd)
        l, t, r, b = self.game_region
        cx = (l + r) // 2
        cy = (t + b) // 2
        mon_idx, mon_origin = monitor_for_point(cx, cy)

        self.capture_backend = cfg.capture_backend
        self.camera = None
        if self.capture_backend in ("windows-capture", "auto") and wc is not None:
            try:
                ml, mt = mon_origin
                region_in_monitor = (l - ml, t - mt, r - ml, b - mt)
                self.camera = WindowsCaptureAdapter(monitor_idx=mon_idx, region_in_monitor=region_in_monitor)
                self.capture_backend = "windows-capture"
            except Exception as e:
                print(f"[capture] windows-capture unavailable, fallback mss: {e}")
                self.camera = None

        if self.camera is None:
            self.camera = MSSCameraAdapter(region=self.game_region, target_fps=90)
            self.camera.start()
            self.capture_backend = "mss"

        self.input = InputDriver(
            hwnd=self.hwnd,
            input_mode=cfg.input_mode,
            allow_physical_fallback=cfg.input_physical_fallback,
            move_physical=cfg.input_move_physical,
        )
        self._auto_scale_ui_config_to_client()

        self.debug_mgr = DebugWindowManager(
            enabled=cfg.debug_windows_enabled,
            target_fps=cfg.debug_render_fps,
        )
        self.debug_windows_enabled = bool(cfg.debug_windows_enabled)

        self.ocr_worker = ShopOcrWorker(lang=cfg.ocr_lang, min_conf=cfg.ocr_min_conf)
        self.ocr_worker.start()
        self.shop_policy = ShopPolicy(
            input_driver=self.input,
            ocr_worker=self.ocr_worker,
            shop_points=cfg.shop_points,
            refresh_rect=cfg.shop_refresh_rect,
            go_rect=cfg.shop_go_rect,
            card_w=cfg.shop_card_w,
            card_h=cfg.shop_card_h,
            card_down=cfg.shop_card_down,
            card_rects=cfg.shop_card_rects,
            buy_min_score=cfg.shop_buy_min_score,
            refresh_max=cfg.shop_refresh_max,
            max_buys=cfg.shop_max_buys,
            action_cooldown_sec=cfg.shop_action_cooldown_sec,
            confirm_frames=cfg.shop_click_confirm_frames,
            confirm_hamming=cfg.shop_click_confirm_hamming,
            buy_retry_max=cfg.shop_buy_retry_max,
            ocr_max_age_sec=cfg.ocr_async_max_age_sec,
            lock_rects=cfg.shop_lock_rects,
            buy_offset_y=cfg.shop_buy_offset_y,
            stick_priority=cfg.shop_stick_priority,
            stick_min_score=cfg.shop_stick_min_score,
            max_weapons=cfg.shop_max_weapons,
            refresh_settle_sec=cfg.shop_refresh_settle_sec,
        )

        self.reward_engine = RewardEngine(
            alive_reward=cfg.reward_alive,
            non_battle_penalty=cfg.reward_non_battle_penalty,
            damage_scale=cfg.reward_damage_scale,
            idle_penalty=cfg.reward_idle_penalty,
            activity_bonus=cfg.reward_activity_bonus,
            loot_bonus=cfg.reward_loot_bonus,
            death_penalty=cfg.reward_death_penalty,
            idle_diff_threshold=cfg.idle_diff_threshold,
            loot_delta_trigger=cfg.loot_delta_trigger,
            time_penalty=cfg.reward_time_penalty,
            kill_bonus=cfg.reward_kill_bonus,
            kill_spawn_trigger=cfg.reward_kill_spawn_trigger,
        )

        # Roboflow 预训练怪物检测器（可选）
        # 启用方式：设置环境变量 ROBOFLOW_API_KEY=your_key
        # 若未设置则回退到原有帧差法
        self.roboflow_detector = None
        if RoboflowMonsterDetector is not None:
            rf_key = os.environ.get("ROBOFLOW_API_KEY", "")
            if rf_key:
                try:
                    self.roboflow_detector = RoboflowMonsterDetector(
                        api_key=rf_key,
                        backend="local",
                        confidence=0.40,
                    )
                except Exception as e:
                    print(f"[roboflow] 检测器加载失败，使用帧差法: {e}")

        self.state_infer = None
        if cfg.use_state_model and YOLO is not None:
            state_path = cfg.state_model_path
            if state_path.strip().lower() == "auto":
                state_path = self._find_latest_state_weight()
            if state_path and os.path.exists(state_path):
                try:
                    self.state_infer = YOLOStateInfer.load(
                        model_path=state_path,
                        image_size=cfg.state_image_size,
                        device=cfg.state_device,
                    )
                    print(f"[state] yolo loaded: {state_path}")
                except Exception as e:
                    print(f"[state] yolo load failed: {e}")

        self.menu_templates = self._load_menu_templates()

        self.upgrade_ocr: Optional[WinMediaOCR] = None
        if cfg.upgrade_ocr_enable:
            try:
                self.upgrade_ocr = WinMediaOCR(lang=cfg.ocr_lang)
                print(f"[upgrade] ocr init ok lang={cfg.ocr_lang}")
            except Exception as e:
                print(f"[upgrade] ocr init failed, fallback to random: {e}")
                self.upgrade_ocr = None

        self.obs_frames: deque[np.ndarray] = deque(maxlen=self.obs_stack)
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_hp = 1.0
        self.last_hp = 1.0
        self.prev_loot_ratio = 0.0
        self.loot_reward_cooldown_left = 0
        self.kill_loot_cooldown_left = 0
        self.loot_collect_min_drop = max(0.002, float(cfg.loot_delta_trigger) * 2.0)
        self.loot_min_prev_ratio = 0.0015
        self.low_hp_streak = 0
        self.death_penalty_applied = False
        self.battle_seen_in_episode = False
        self.last_battle_action = 0
        self.same_action_streak = 0
        self.low_motion_streak = 0
        self.stuck_break_left = 0
        self.stuck_break_count = 0
        self.zero_action_streak = 0
        self.zero_action_grace_frames = 3

        self.state_name = "unknown"
        self.state_score = 0.0
        self.raw_state = "unknown"
        self.raw_state_score = 0.0
        self.last_tpl_scores = {"go": 0.0, "choose": 0.0, "restart": 0.0}

        self.control_armed = not cfg.require_arm_hotkey
        self._arm_hotkey_down = False
        self._debug_hotkey_down = False
        self._quit_hotkey_down = False
        self._arm_hotkey_last_ts = 0.0
        self._debug_hotkey_last_ts = 0.0
        self._quit_hotkey_last_ts = 0.0
        self._ui_action_last_ts = 0.0
        self.last_menu_action_ts = 0.0
        self._upgrade_menu_streak = 0
        self._upgrade_guard_until_ts = 0.0
        self._upgrade_last_pick_idx = -1
        self.non_battle_hold_until_ts = 0.0
        self.shop_enter_ts = 0.0
        self.shop_lock_until_ts = 0.0
        self._shop_soft_until_ts = 0.0
        self._in_shop_prev = False
        self._control_panel_cb_installed = False
        self._btn_arm = (12, 14, 176, 52)
        self._btn_debug = (188, 14, 352, 52)
        self._btn_stop = (364, 14, 528, 52)

        self.align_until_ts = time.time() + float(max(0.0, cfg.align_wait_sec))
        self._control_panel_last_ts = 0.0
        self._control_panel_interval = 1.0 / 15.0  # render at most 15fps

        self.step_counter = 0
        self.episode_reward = 0.0
        self.total_reward = 0.0
        self.last_info: Dict[str, object] = {}
        self.last_step_ts = time.time()
        self.fps_hist: deque[float] = deque(maxlen=120)

        # State inference caching: skip heavy YOLO+template during battle
        self._state_infer_interval = max(1, int(cfg.state_infer_interval))
        self._state_infer_counter = 0
        self._cached_state: Tuple[str, float, float, float, float] = ("unknown", 0.0, 0.0, 0.0, 0.0)

        print(f"[capture] backend={self.capture_backend} region={self.game_region}")
        print("[control] press F7 to start/pause automation")
        print("[control] press F8 to stop training")
        print("[debug] press F6 to show/hide debug windows")
        print(
            "[battle] anti-stuck="
            f"{'on' if self.cfg.anti_stuck_enable else 'off'} "
            f"same>={self.cfg.anti_stuck_same_action_steps} "
            f"low_motion>={self.cfg.anti_stuck_low_motion_steps} "
            f"break={self.cfg.anti_stuck_break_steps}"
        )

    @staticmethod
    def _scale_point(pt: Tuple[int, int], sx: float, sy: float, client_w: int, client_h: int) -> Tuple[int, int]:
        x = int(round(float(pt[0]) * float(sx)))
        y = int(round(float(pt[1]) * float(sy)))
        x = int(np.clip(x, 0, max(0, int(client_w) - 1)))
        y = int(np.clip(y, 0, max(0, int(client_h) - 1)))
        return (x, y)

    @staticmethod
    def _scale_rect(
        rect: Tuple[int, int, int, int],
        sx: float,
        sy: float,
        client_w: int,
        client_h: int,
    ) -> Tuple[int, int, int, int]:
        x1 = int(round(float(rect[0]) * float(sx)))
        y1 = int(round(float(rect[1]) * float(sy)))
        x2 = int(round(float(rect[2]) * float(sx)))
        y2 = int(round(float(rect[3]) * float(sy)))
        lx, rx = sorted((x1, x2))
        ty, by = sorted((y1, y2))
        lx = int(np.clip(lx, 0, max(0, int(client_w) - 1)))
        rx = int(np.clip(rx, 0, max(0, int(client_w) - 1)))
        ty = int(np.clip(ty, 0, max(0, int(client_h) - 1)))
        by = int(np.clip(by, 0, max(0, int(client_h) - 1)))
        if rx < lx:
            rx = lx
        if by < ty:
            by = ty
        return (lx, ty, rx, by)

    @staticmethod
    def _guess_ui_base_resolution(max_x: int, max_y: int) -> Tuple[int, int]:
        # Heuristic: legacy presets are mainly 1920x1080 or 1280x720.
        if int(max_x) >= 1700 or int(max_y) >= 980:
            return (1920, 1080)
        if int(max_x) >= 1180 or int(max_y) >= 680:
            return (1280, 720)
        return (max(1, int(max_x) + 1), max(1, int(max_y) + 1))

    def _auto_scale_ui_config_to_client(self):
        l, t, r, b = self.game_region
        client_w = max(1, int(r - l))
        client_h = max(1, int(b - t))

        pts: List[Tuple[int, int]] = []
        pts.extend(list(self.cfg.shop_points or []))
        pts.extend(list(self.cfg.upgrade_select_points or []))
        if self.cfg.gameover_click_pos is not None:
            pts.append(tuple(self.cfg.gameover_click_pos))

        rects: List[Tuple[int, int, int, int]] = [
            tuple(self.cfg.shop_refresh_rect),
            tuple(self.cfg.shop_go_rect),
            tuple(self.cfg.item_pick_take_rect),
            tuple(self.cfg.item_pick_recycle_rect),
            tuple(self.cfg.gameover_restart_rect),
            tuple(self.cfg.gameover_new_game_rect),
        ]
        rects.extend(list(self.cfg.shop_card_rects or []))
        rects.extend(list(self.cfg.shop_lock_rects or []))
        if self.cfg.upgrade_select_rect is not None:
            rects.append(tuple(self.cfg.upgrade_select_rect))
        if self.cfg.upgrade_refresh_avoid_rect is not None:
            rects.append(tuple(self.cfg.upgrade_refresh_avoid_rect))

        max_x = 0
        max_y = 0
        for px, py in pts:
            max_x = max(max_x, int(px))
            max_y = max(max_y, int(py))
        for x1, y1, x2, y2 in rects:
            max_x = max(max_x, int(x1), int(x2))
            max_y = max(max_y, int(y1), int(y2))

        src_w, src_h = self._guess_ui_base_resolution(max_x=max_x, max_y=max_y)
        overflow = bool(max_x >= client_w or max_y >= client_h)
        if overflow and (src_w > 0 and src_h > 0):
            sx = float(client_w) / float(src_w)
            sy = float(client_h) / float(src_h)
        else:
            sx = 1.0
            sy = 1.0

        old_shop_points = list(self.cfg.shop_points)
        old_upgrade_points = list(self.cfg.upgrade_select_points)
        old_gameover_pos = self.cfg.gameover_click_pos
        old_shop_refresh_rect = tuple(self.cfg.shop_refresh_rect)
        old_shop_go_rect = tuple(self.cfg.shop_go_rect)
        old_item_pick_take_rect = tuple(self.cfg.item_pick_take_rect)
        old_item_pick_recycle_rect = tuple(self.cfg.item_pick_recycle_rect)
        old_gameover_restart_rect = tuple(self.cfg.gameover_restart_rect)
        old_gameover_new_game_rect = tuple(self.cfg.gameover_new_game_rect)
        old_shop_lock_rects = list(self.cfg.shop_lock_rects)
        old_shop_card_rects = list(self.cfg.shop_card_rects)
        old_upgrade_select_rect = self.cfg.upgrade_select_rect
        old_upgrade_refresh_avoid_rect = self.cfg.upgrade_refresh_avoid_rect

        self.cfg.shop_points = [
            self._scale_point(tuple(pt), sx=sx, sy=sy, client_w=client_w, client_h=client_h)
            for pt in list(self.cfg.shop_points or [])
        ]
        self.cfg.upgrade_select_points = [
            self._scale_point(tuple(pt), sx=sx, sy=sy, client_w=client_w, client_h=client_h)
            for pt in list(self.cfg.upgrade_select_points or [])
        ]
        self.cfg.shop_refresh_rect = self._scale_rect(
            tuple(self.cfg.shop_refresh_rect),
            sx=sx,
            sy=sy,
            client_w=client_w,
            client_h=client_h,
        )
        self.cfg.shop_go_rect = self._scale_rect(
            tuple(self.cfg.shop_go_rect),
            sx=sx,
            sy=sy,
            client_w=client_w,
            client_h=client_h,
        )
        self.cfg.item_pick_take_rect = self._scale_rect(
            tuple(self.cfg.item_pick_take_rect),
            sx=sx,
            sy=sy,
            client_w=client_w,
            client_h=client_h,
        )
        self.cfg.item_pick_recycle_rect = self._scale_rect(
            tuple(self.cfg.item_pick_recycle_rect),
            sx=sx,
            sy=sy,
            client_w=client_w,
            client_h=client_h,
        )
        self.cfg.gameover_restart_rect = self._scale_rect(
            tuple(self.cfg.gameover_restart_rect),
            sx=sx,
            sy=sy,
            client_w=client_w,
            client_h=client_h,
        )
        self.cfg.gameover_new_game_rect = self._scale_rect(
            tuple(self.cfg.gameover_new_game_rect),
            sx=sx,
            sy=sy,
            client_w=client_w,
            client_h=client_h,
        )
        self.cfg.shop_lock_rects = [
            self._scale_rect(tuple(rc), sx=sx, sy=sy, client_w=client_w, client_h=client_h)
            for rc in list(self.cfg.shop_lock_rects or [])
        ]
        self.cfg.shop_card_rects = [
            self._scale_rect(tuple(rc), sx=sx, sy=sy, client_w=client_w, client_h=client_h)
            for rc in list(self.cfg.shop_card_rects or [])
        ]
        if self.cfg.upgrade_select_rect is not None:
            self.cfg.upgrade_select_rect = self._scale_rect(
                tuple(self.cfg.upgrade_select_rect),
                sx=sx,
                sy=sy,
                client_w=client_w,
                client_h=client_h,
            )
        if self.cfg.upgrade_refresh_avoid_rect is not None:
            self.cfg.upgrade_refresh_avoid_rect = self._scale_rect(
                tuple(self.cfg.upgrade_refresh_avoid_rect),
                sx=sx,
                sy=sy,
                client_w=client_w,
                client_h=client_h,
            )
        if self.cfg.gameover_click_pos is not None:
            self.cfg.gameover_click_pos = self._scale_point(
                tuple(self.cfg.gameover_click_pos),
                sx=sx,
                sy=sy,
                client_w=client_w,
                client_h=client_h,
            )

        changed = bool(
            old_shop_points != self.cfg.shop_points
            or old_upgrade_points != self.cfg.upgrade_select_points
            or old_gameover_pos != self.cfg.gameover_click_pos
            or old_shop_refresh_rect != self.cfg.shop_refresh_rect
            or old_shop_go_rect != self.cfg.shop_go_rect
            or old_item_pick_take_rect != self.cfg.item_pick_take_rect
            or old_item_pick_recycle_rect != self.cfg.item_pick_recycle_rect
            or old_gameover_restart_rect != self.cfg.gameover_restart_rect
            or old_gameover_new_game_rect != self.cfg.gameover_new_game_rect
            or old_shop_lock_rects != self.cfg.shop_lock_rects
            or old_shop_card_rects != self.cfg.shop_card_rects
            or old_upgrade_select_rect != self.cfg.upgrade_select_rect
            or old_upgrade_refresh_avoid_rect != self.cfg.upgrade_refresh_avoid_rect
        )
        if changed:
            print(
                f"[config] ui coords adjusted src=({src_w}x{src_h}) dst=({client_w}x{client_h}) "
                f"scale=({sx:.3f},{sy:.3f})"
            )

    def _reset_battle_action_trackers(self):
        self.last_battle_action = 0
        self.same_action_streak = 0
        self.low_motion_streak = 0
        self.stuck_break_left = 0
        self.zero_action_streak = 0

    @staticmethod
    def _anti_stuck_pick_alternative(action: int) -> int:
        # Prefer turn + reverse + brief stop, which is better than random same-axis spam.
        a = int(action)
        if a == 1:  # up
            cands = [3, 4, 2, 0]
        elif a == 2:  # down
            cands = [3, 4, 1, 0]
        elif a == 3:  # left
            cands = [1, 2, 4, 0]
        elif a == 4:  # right
            cands = [1, 2, 3, 0]
        else:
            cands = [0, 1, 2, 3, 4]
        return int(random.choice(cands))

    def _find_latest_state_weight(self) -> str:
        roots = self._resource_roots()
        cands: List[Path] = []
        for root in roots:
            runs_dir = root / "runs" / "classify"
            if not runs_dir.exists():
                pass
            else:
                cands.extend(list(runs_dir.glob("*/weights/best.onnx")))
                cands.extend(list(runs_dir.glob("*/weights/best.pt")))
            classifier_dir = root / "classifier"
            if classifier_dir.exists():
                cands.extend(list(classifier_dir.glob("**/best.onnx")))
                cands.extend(list(classifier_dir.glob("**/best.pt")))
        if not cands:
            return ""
        best = max(cands, key=lambda p: p.stat().st_mtime)
        return str(best)

    def _load_menu_templates(self) -> Dict[str, List[np.ndarray]]:
        bases = [root / "assets" for root in self._resource_roots()]
        spec = {
            "GO": ["go.png", "d_go.png"],
            "CHOOSE": ["chose.png", "d_chose.png"],
            "RESTART": ["restart.png", "d_restard.png"],
        }
        out: Dict[str, List[np.ndarray]] = {}
        for key, files in spec.items():
            items: List[np.ndarray] = []
            for name in files:
                for base in bases:
                    p = base / name
                    if p.exists():
                        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        if img is not None and img.size > 0:
                            items.append(img)
                            break
            out[key] = items
        return out

    def _resource_roots(self) -> List[Path]:
        here = Path(__file__).resolve().parent.parent
        roots: List[Path] = []
        raw_dir_cfg = str(getattr(self.cfg, "raw_models_dir", "") or "").strip()
        if raw_dir_cfg:
            raw_dir = Path(raw_dir_cfg)
            if not raw_dir.is_absolute():
                raw_dir = (here.parent / raw_dir).resolve()
            if raw_dir.exists() and raw_dir.is_dir():
                roots.append(raw_dir)
        roots.append(here)
        sibling_health = here.parent / "health"
        if sibling_health.exists() and sibling_health.is_dir() and sibling_health != here:
            roots.append(sibling_health)
        uniq: List[Path] = []
        seen = set()
        for p in roots:
            sp = str(p.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            uniq.append(p)
        return uniq

    def _blank_frame(self) -> np.ndarray:
        l, t, r, b = self.game_region
        w = max(1, r - l)
        h = max(1, b - t)
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _grab_frame(self) -> np.ndarray:
        try:
            latest = self.camera.get_latest_frame()
            if latest is not None and latest.size > 0:
                return latest
        except Exception:
            pass
        return self._blank_frame()

    def _poll_hotkey_pressed(self, vk: int, attr_name: str) -> bool:
        try:
            state = int(ctypes.windll.user32.GetAsyncKeyState(vk))
            is_down = bool(state & 0x8000)
        except Exception:
            return False
        was_down = bool(getattr(self, attr_name, False))
        setattr(self, attr_name, is_down)
        # Use rising-edge only. The low-bit "pressed since last call" can
        # retrigger under key-repeat and flip armed/pause unexpectedly.
        return bool(is_down and (not was_down))

    def _poll_hotkeys(self):
        now = time.time()
        if self._poll_hotkey_pressed(0x76, "_arm_hotkey_down"):
            if now - self._arm_hotkey_last_ts >= self.cfg.hotkey_debounce_sec:
                self._arm_hotkey_last_ts = now
                self.control_armed = not self.control_armed
                if not self.control_armed:
                    self.input.release_movement()
                else:
                    # Arm immediately after menu operations.
                    self.non_battle_hold_until_ts = 0.0
                    if bool(getattr(self.cfg, "input_move_physical", False)):
                        self._focus_game_window_soft()
                print(f"[control] {'armed' if self.control_armed else 'paused'} (F7)")

        if self._poll_hotkey_pressed(0x75, "_debug_hotkey_down"):
            if now - self._debug_hotkey_last_ts >= self.cfg.hotkey_debounce_sec:
                self._debug_hotkey_last_ts = now
                enabled = self.debug_mgr.toggle()
                self.debug_windows_enabled = enabled
                print(f"[debug] {'shown' if enabled else 'hidden'} (F6)")

        if self.cfg.enable_quit_hotkey and self._poll_hotkey_pressed(0x77, "_quit_hotkey_down"):
            if now - self._quit_hotkey_last_ts >= self.cfg.hotkey_debounce_sec:
                self._quit_hotkey_last_ts = now
                if self.stop_manager.request_stop("F8"):
                    print("[control] stop requested (F8)")

    def _focus_game_window_soft(self):
        try:
            ctypes.windll.user32.ShowWindow(self.hwnd, 5)
            ctypes.windll.user32.SetForegroundWindow(self.hwnd)
            ctypes.windll.user32.SetActiveWindow(self.hwnd)
        except Exception:
            pass

    @staticmethod
    def _in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return (x1 <= int(x) <= x2) and (y1 <= int(y) <= y2)

    def _on_control_panel_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        now = time.time()
        if (now - self._ui_action_last_ts) < float(self.cfg.hotkey_debounce_sec):
            return
        self._ui_action_last_ts = now

        if self._in_rect(x, y, self._btn_arm):
            self.control_armed = not self.control_armed
            if not self.control_armed:
                self.input.release_movement()
            else:
                self.non_battle_hold_until_ts = 0.0
                if bool(getattr(self.cfg, "input_move_physical", False)):
                    self._focus_game_window_soft()
            print(f"[control] {'armed' if self.control_armed else 'paused'} (panel)")
            return

        if self._in_rect(x, y, self._btn_debug):
            enabled = self.debug_mgr.toggle()
            self.debug_windows_enabled = enabled
            print(f"[debug] {'shown' if enabled else 'hidden'} (panel)")
            return

        if self._in_rect(x, y, self._btn_stop):
            if self.stop_manager.request_stop("panel_stop"):
                print("[control] stop requested (panel)")

    def _render_control_panel(self):
        now = time.perf_counter()
        if now - self._control_panel_last_ts < self._control_panel_interval:
            return
        self._control_panel_last_ts = now

        panel = np.zeros((66, 540, 3), dtype=np.uint8)
        panel[:, :] = (24, 24, 24)
        cv2.putText(panel, "Control", (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        arm_color = (0, 140, 0) if self.control_armed else (60, 60, 60)
        dbg_color = (130, 80, 0) if self.debug_windows_enabled else (60, 60, 60)
        stop_color = (0, 0, 160)
        for rect, color in ((self._btn_arm, arm_color), (self._btn_debug, dbg_color), (self._btn_stop, stop_color)):
            x1, y1, x2, y2 = rect
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (180, 180, 180), 1)

        cv2.putText(panel, "Start/Pause", (26, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1)
        cv2.putText(panel, "Debug On/Off", (198, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1)
        cv2.putText(panel, "Stop", (427, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

        try:
            cv2.imshow(CONTROL_PANEL_WINDOW, panel)
            if not self._control_panel_cb_installed:
                cv2.setMouseCallback(CONTROL_PANEL_WINDOW, self._on_control_panel_mouse)
                self._control_panel_cb_installed = True
            cv2.waitKey(1)
        except Exception:
            pass

    def _normalize_state_name(self, state_name: str) -> str:
        s = str(state_name or "").strip().lower()
        if not s:
            return "unknown"
        direct = {
            "combat": "battle",
            "fight": "battle",
            "wave": "battle",
            "go": "shop",
            "store": "shop",
            "merchant": "shop",
            "choose": "upgrade",
            "levelup": "upgrade",
            "level_up": "upgrade",
            "item_pick": "item_pick",
            "item-pick": "item_pick",
            "itempick": "item_pick",
            "pick": "item_pick",
            "reward": "item_pick",
            "restart": "gameover",
            "game_over": "gameover",
            "game-over": "gameover",
            "death": "gameover",
            "dead": "gameover",
            "defeat": "gameover",
            "lose": "gameover",
            "lost": "gameover",
        }
        if s in direct:
            return direct[s]
        squashed = s.replace(" ", "").replace("-", "").replace("_", "")
        squash_map = {
            "battle": "battle",
            "shop": "shop",
            "upgrade": "upgrade",
            "itempick": "item_pick",
            "gameover": "gameover",
            "unknown": "unknown",
        }
        return squash_map.get(squashed, s)

    def _fallback_menu_state(
        self,
        state_name: str,
        state_score: float,
        go_tpl: float,
        choose_tpl: float,
        restart_tpl: float,
    ) -> str:
        s = self._normalize_state_name(state_name)
        if s == "battle" and float(state_score) >= float(self.cfg.state_non_battle_min_score):
            return "battle"
        if s == "shop" and go_tpl < 0.45 and max(choose_tpl, restart_tpl) < 0.45:
            return "unknown"
        if s not in ("unknown", "battle"):
            return s
        if restart_tpl >= 0.56:
            return "gameover"
        if choose_tpl >= 0.60:
            return "upgrade"
        if go_tpl >= 0.66 or (s == "unknown" and go_tpl >= 0.58):
            return "shop"
        return s

    def _script_menu_state(self, state_name: str, choose_tpl: float, restart_tpl: float) -> str:
        s = self._normalize_state_name(state_name)
        # Non-shop/non-battle interfaces are script-driven.
        if s in ("gameover", "item_pick", "upgrade"):
            return s
        if restart_tpl >= 0.48:
            return "gameover"
        if choose_tpl >= 0.50:
            return "upgrade"
        return ""

    def _infer_state_with_templates(self, frame: np.ndarray) -> Tuple[str, float, float, float, float]:
        """Returns (state_name, state_score, go_tpl, choose_tpl, restart_tpl).

        Uses caching: during confirmed battle, full inference is only
        performed every ``_state_infer_interval`` steps.  On cache-hit
        steps the previous result is returned, saving ~35ms per step.
        """
        if frame is None or frame.size == 0:
            return ("unknown", 0.0, 0.0, 0.0, 0.0)

        # Check if we can reuse cached state (battle fast-path)
        self._state_infer_counter += 1
        cached_state_name = self._cached_state[0]
        if (
            cached_state_name == "battle"
            and self._state_infer_counter < self._state_infer_interval
        ):
            return self._cached_state

        self._state_infer_counter = 0

        # Convert to BGR once — reused by templates and YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Compute template scores using pre-converted BGR frame
        go_tpl = self._template_best_score_bgr(frame_bgr, "GO")
        choose_tpl = self._template_best_score_bgr(frame_bgr, "CHOOSE")
        restart_tpl = self._template_best_score_bgr(frame_bgr, "RESTART")

        # Template-first menu checks as robust guard.
        if restart_tpl >= 0.58:
            result = ("gameover", 0.98, go_tpl, choose_tpl, restart_tpl)
            self._cached_state = result
            return result
        if choose_tpl >= 0.70:
            result = ("upgrade", 0.96, go_tpl, choose_tpl, restart_tpl)
            self._cached_state = result
            return result
        if go_tpl >= 0.72:
            result = ("shop", 0.95, go_tpl, choose_tpl, restart_tpl)
            self._cached_state = result
            return result

        if self.state_infer is None:
            result = ("unknown", 0.0, go_tpl, choose_tpl, restart_tpl)
            self._cached_state = result
            return result

        try:
            top1, p1, _top2, _p2 = self.state_infer.predict_top2(frame_bgr)
            s = self._normalize_state_name(top1)
            result = (s, float(p1), go_tpl, choose_tpl, restart_tpl)
        except Exception:
            result = ("unknown", 0.0, go_tpl, choose_tpl, restart_tpl)
        self._cached_state = result
        return result

    @staticmethod
    def _clip_rect_to_frame(rect: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = [int(v) for v in rect]
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
        y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _point_in_rect(pt: Tuple[int, int], rect: Optional[Tuple[int, int, int, int]]) -> bool:
        if rect is None:
            return False
        x, y = int(pt[0]), int(pt[1])
        x1, y1, x2, y2 = [int(v) for v in rect]
        return bool(x1 <= x <= x2 and y1 <= y <= y2)

    @staticmethod
    def _rect_intersects(a: Tuple[int, int, int, int], b: Optional[Tuple[int, int, int, int]]) -> bool:
        if b is None:
            return False
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _template_search_rect(self, key: str, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        k = str(key or "").upper()
        if k == "GO":
            x1, y1, x2, y2 = self.cfg.shop_go_rect
            return self._clip_rect_to_frame((x1 - 28, y1 - 28, x2 + 28, y2 + 28), w, h)
        if k == "RESTART":
            if self.cfg.gameover_click_pos is not None:
                cx, cy = self.cfg.gameover_click_pos
                return self._clip_rect_to_frame((cx - 280, cy - 140, cx + 280, cy + 140), w, h)
            x1, y1, x2, y2 = self.cfg.gameover_restart_rect
            return self._clip_rect_to_frame((x1 - 36, y1 - 36, x2 + 36, y2 + 36), w, h)
        if k == "CHOOSE":
            if self.cfg.upgrade_select_rect is not None:
                x1, y1, x2, y2 = self.cfg.upgrade_select_rect
                return self._clip_rect_to_frame((x1 - 40, y1 - 40, x2 + 40, y2 + 40), w, h)
            pts = list(self.cfg.upgrade_select_points or [])
            if pts:
                xs = [int(p[0]) for p in pts]
                ys = [int(p[1]) for p in pts]
                return self._clip_rect_to_frame(
                    (min(xs) - 300, min(ys) - 260, max(xs) + 300, max(ys) + 180),
                    w,
                    h,
                )
            x1, y1, x2, y2 = self.cfg.item_pick_take_rect
            return self._clip_rect_to_frame((x1 - 42, y1 - 42, x2 + 42, y2 + 42), w, h)
        return None

    def _template_best_score(self, frame_rgb: np.ndarray, key: str) -> float:
        """Convenience wrapper: converts RGB→BGR then delegates."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self._template_best_score_bgr(frame_bgr, key)

    def _template_best_score_bgr(self, frame_bgr: np.ndarray, key: str) -> float:
        """Template match on a pre-converted BGR frame (avoids redundant cvtColor)."""
        templates = self.menu_templates.get(key, [])
        if len(templates) == 0:
            return 0.0
        search_rect = self._template_search_rect(key, frame_bgr)
        if search_rect is not None:
            x1, y1, x2, y2 = search_rect
            search_img = frame_bgr[y1:y2, x1:x2]
        else:
            search_img = frame_bgr
        best = -1.0
        for tpl in templates:
            th, tw = tpl.shape[:2]
            fh, fw = search_img.shape[:2]
            if fh < th or fw < tw:
                continue
            res = cv2.matchTemplate(search_img, tpl, cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, _maxl = cv2.minMaxLoc(res)
            best = max(best, float(maxv))
        return max(0.0, best)

    def _is_template_hit(self, frame_rgb: np.ndarray, key: str, threshold: float) -> bool:
        if frame_rgb is None or frame_rgb.size == 0:
            return False
        return self._template_best_score(frame_rgb, key) >= float(threshold)

    def _menu_action_ready(self) -> bool:
        now = time.time()
        if now - float(self.last_menu_action_ts) < float(self.cfg.menu_action_cooldown_sec):
            return False
        self.last_menu_action_ts = now
        return True

    def _upgrade_ocr_pick(self) -> Optional[Tuple[bool, str]]:
        """Use OCR to pick the best upgrade option for the savage/primitive build.
        Returns (ok, msg) if OCR succeeds, or None to fall back to random."""
        if self.upgrade_ocr is None:
            return None
        pts = list(self.cfg.upgrade_select_points or [])
        avoid_rect = self.cfg.upgrade_refresh_avoid_rect
        if avoid_rect is not None:
            pts = [p for p in pts if not self._point_in_rect(tuple(p), avoid_rect)]
        if not pts:
            return None

        frame = self._grab_frame()
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        card_w = int(self.cfg.upgrade_card_w)
        card_h = int(self.cfg.upgrade_card_h)

        scored: List[Tuple[int, float, str]] = []  # (idx, score, debug_text)
        for idx, pt in enumerate(pts):
            cx, cy = int(pt[0]), int(pt[1])
            x1 = int(np.clip(cx - card_w // 2, 0, max(0, w - 1)))
            x2 = int(np.clip(cx + card_w // 2, x1 + 1, max(x1 + 1, w)))
            y1 = int(np.clip(cy - card_h, 0, max(0, h - 1)))
            y2 = int(np.clip(cy + card_h // 4, y1 + 1, max(y1 + 1, h)))
            crop_rgb = frame[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                scored.append((idx, 0.0, "empty"))
                continue
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            try:
                lines = self.upgrade_ocr.predict(crop_bgr)
            except Exception:
                scored.append((idx, 0.0, "ocr_fail"))
                continue
            total_score = 0.0
            texts = []
            for txt, conf in lines:
                total_score += score_upgrade_text(txt, conf, float(self.cfg.ocr_min_conf))
                texts.append(normalize_text(txt)[:20])
            scored.append((idx, total_score, "|".join(texts[:3])))

        if not scored:
            return None

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score, best_text = scored[0]

        if best_score <= 0.0:
            # No meaningful match, fall back to random
            return None

        pt = pts[best_idx]
        click_res = self.input.click_client_point(pt)
        debug_info = f"upgrade_ocr_pick:{best_idx} score={best_score:.2f} text={best_text} click={click_res.method}:{click_res.ok}"
        print(f"[upgrade] {debug_info}")
        return bool(click_res.ok), debug_info

    def _menu_interact(self, state_name: str) -> Tuple[bool, str]:
        s = self._normalize_state_name(state_name)
        if not self._menu_action_ready():
            return False, "cooldown"
        self._focus_game_window_soft()

        if s in ("shop", "go"):
            click_res = self.input.click_client_rect(self.cfg.shop_go_rect)
            return bool(click_res.ok), f"shop_go:{click_res.method}:{click_res.ok}"

        if s in ("upgrade", "choose", "item_pick"):
            if s == "item_pick":
                click_res = self.input.click_client_rect(self.cfg.item_pick_take_rect)
                return bool(click_res.ok), f"item_pick_take:{click_res.method}:{click_res.ok}"

            # Prevent infinite reroll loops on upgrade screen.
            now = time.time()
            if now < float(self._upgrade_guard_until_ts):
                left = max(0.0, float(self._upgrade_guard_until_ts) - now)
                return False, f"upgrade_guard_cooldown:{left:.2f}s"
            if int(self._upgrade_menu_streak) >= 10:
                self._upgrade_guard_until_ts = now + 1.20
                self._upgrade_menu_streak = 0
                self._upgrade_last_pick_idx = -1
                return False, "upgrade_guard_triggered"

            # Try OCR-based intelligent upgrade selection for savage/primitive build
            tried: List[str] = []
            ocr_picked = self._upgrade_ocr_pick()
            if ocr_picked is not None:
                ocr_ok, ocr_msg = ocr_picked
                tried.append(f"ocr:{ocr_msg}")
                if bool(ocr_ok):
                    if bool(self.cfg.upgrade_press_enter_confirm):
                        self.input.press_key("enter")
                    self._upgrade_last_pick_idx = -1
                    return True, f"{ocr_msg}|confirm={bool(self.cfg.upgrade_press_enter_confirm)}"

            # Fallback: prefer explicit card points. Rect click is last resort
            # because a broad rect may include the reroll button.
            if len(self.cfg.upgrade_select_points) > 0:
                order_raw = [int(i) for i in np.random.permutation(len(self.cfg.upgrade_select_points))]
                if (
                    len(order_raw) > 1
                    and int(self._upgrade_last_pick_idx) in order_raw
                ):
                    last_idx = int(self._upgrade_last_pick_idx)
                    order_raw = [i for i in order_raw if i != last_idx] + [last_idx]
                for raw_idx in order_raw:
                    idx = int(raw_idx)
                    pt = self.cfg.upgrade_select_points[idx]
                    if self._point_in_rect(tuple(pt), self.cfg.upgrade_refresh_avoid_rect):
                        tried.append(f"pt{idx}:skip_avoid_rect")
                        continue
                    click_res = self.input.click_client_point(pt)
                    tried.append(f"pt{idx}:{click_res.method}:{click_res.ok}")
                    if bool(click_res.ok):
                        if bool(self.cfg.upgrade_press_enter_confirm):
                            self.input.press_key("enter")
                        self._upgrade_last_pick_idx = idx
                        return True, (
                            f"upgrade_pick:{idx}:{click_res.method}:{click_res.ok}"
                            f"|confirm={bool(self.cfg.upgrade_press_enter_confirm)}"
                        )
            if self.cfg.upgrade_select_rect is not None:
                if self._rect_intersects(
                    tuple(self.cfg.upgrade_select_rect),
                    self.cfg.upgrade_refresh_avoid_rect,
                ):
                    tried.append("rect:skip_overlap_avoid")
                else:
                    click_res = self.input.click_client_rect(self.cfg.upgrade_select_rect)
                    tried.append(f"rect:{click_res.method}:{click_res.ok}")
                    if bool(click_res.ok):
                        if bool(self.cfg.upgrade_press_enter_confirm):
                            self.input.press_key("enter")
                        self._upgrade_last_pick_idx = -1
                        return True, (
                            f"upgrade_pick_rect:{click_res.method}:{click_res.ok}"
                            f"|confirm={bool(self.cfg.upgrade_press_enter_confirm)}"
                        )
            tried_msg = ",".join(tried[:4]) if tried else "no_click_target"
            return False, f"upgrade_no_click|fallback_after={tried_msg}"

        if s in ("gameover", "restart", "game_over"):
            tried: List[str] = []
            if self.cfg.gameover_click_pos is not None:
                click_res = self.input.click_client_point(self.cfg.gameover_click_pos)
                tried.append(f"pos:{click_res.method}:{click_res.ok}")
                if bool(click_res.ok):
                    return True, f"gameover_click_pos:{click_res.method}:{click_res.ok}"
            click_res = self.input.click_client_rect(self.cfg.gameover_restart_rect)
            tried.append(f"rect:{click_res.method}:{click_res.ok}")
            return bool(click_res.ok), f"gameover_restart_rect:{click_res.method}:{click_res.ok}|fallback_from={','.join(tried[:3])}"

        return False, "no_menu_action"

    def _predict_hp(self, frame_rgb: np.ndarray) -> float:
        x1, y1, x2, y2 = self.cfg.hp_rect
        h, w = frame_rgb.shape[:2]
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
        y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return float(np.clip(self.last_hp, 0.0, 1.0))
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        m1 = cv2.inRange(hsv, (0, 90, 90), (12, 255, 255))
        m2 = cv2.inRange(hsv, (170, 90, 90), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
        col = np.max(mask, axis=0)
        ratio = float(np.count_nonzero(col > 0)) / float(max(1, col.shape[0]))
        return float(np.clip(ratio, 0.0, 1.0))

    def _extract_danger_channel(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Extract a danger-highlight channel from the RGB frame.

        Detects red/orange projectiles and bright enemy indicators via
        HSV thresholding.  Returns a single-channel uint8 image at the
        original frame resolution (caller resizes).
        """
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        # Red bullets / damage indicators (hue wraps around 0)
        red_lo = cv2.inRange(hsv, (0, 80, 100), (15, 255, 255))
        red_hi = cv2.inRange(hsv, (165, 80, 100), (180, 255, 255))
        # Orange/yellow projectiles
        orange = cv2.inRange(hsv, (15, 90, 120), (30, 255, 255))
        # Bright magenta/purple enemy auras
        purple = cv2.inRange(hsv, (130, 60, 100), (165, 255, 255))
        danger = cv2.bitwise_or(red_lo, red_hi)
        danger = cv2.bitwise_or(danger, orange)
        danger = cv2.bitwise_or(danger, purple)
        # Slight dilation to make small projectiles more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        danger = cv2.dilate(danger, kernel, iterations=1)
        return danger

    def _build_semantic_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Build a clean semantic mask for CNN input.

        Returns an (obs_size, obs_size, 3) uint8 RGB image on a black canvas:
          R channel: moving objects with large area = enemy bodies
          G channel: loot / healing items (green hues)
          B channel: player self, always drawn at image center
          White (R+G+B=255): moving objects with small area = bullets/projectiles

        Enemy detection uses frame-diff motion rather than colour, so it works
        for any enemy sprite regardless of its appearance.  Large motion blobs
        are enemies; small fast blobs are bullets.
        """
        sz = self.obs_size
        frame_small = cv2.resize(frame_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_RGB2HSV)

        # Black canvas (H, W, 3) in RGB
        mask = np.zeros((sz, sz, 3), dtype=np.uint8)

        # --- Motion-based detection (enemies + bullets) ---
        # Bullet filter: projectiles are bright & saturated; floor decorations are dull.
        # HSV saturation channel (index 1) and value/brightness channel (index 2).
        sat = hsv[:, :, 1]   # 0-255, high = vivid colour
        val = hsv[:, :, 2]   # 0-255, high = bright
        # A pixel qualifies as a potential bullet if it is bright OR highly saturated.
        bullet_color_mask = ((sat > 90) | (val > 200)).astype(np.uint8) * 255

        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)

        if self.roboflow_detector is not None:
            # --- Roboflow 多类别检测：怪物、子弹、掉落物、玩家、建筑全部由模型处理 ---
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            channels = self.roboflow_detector.build_channels(frame_bgr, sz)
            mask[:, :, 0] = channels["enemy"]   # Red   = 怪物
            mask[:, :, 1] = channels["loot"]    # Green = 掉落物
            # 子弹：叠加到三个通道使其显示为白色点（R+G+B=255）
            bullet_ch = channels["bullet"]
            mask[:, :, 0] = np.maximum(mask[:, :, 0], bullet_ch)
            mask[:, :, 1] = np.maximum(mask[:, :, 1], bullet_ch)
            mask[:, :, 2] = np.maximum(mask[:, :, 2], bullet_ch)
            # 建筑：低亮度轮廓叠加到蓝通道（障碍物提示，不覆盖玩家实心圆）
            mask[:, :, 2] = np.maximum(mask[:, :, 2], channels["structure"])
            # 玩家：若模型检测到 'you'，缓存位置供蓝通道使用
            if channels["player_pos"] is not None:
                self._last_player_pos = channels["player_pos"]
        else:
            # --- 原有帧差法（Roboflow 不可用时回退） ---
            if self._prev_gray is not None and self._prev_gray.shape == gray_small.shape:
                raw_motion = cv2.absdiff(gray_small, self._prev_gray)
                _, motion_thresh = cv2.threshold(raw_motion, 25, 255, cv2.THRESH_BINARY)
                # Slight dilation to merge fragmented blobs from a single enemy
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                motion_thresh = cv2.dilate(motion_thresh, kernel, iterations=1)
                motion_contours, _ = cv2.findContours(
                    motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in motion_contours:
                    area = cv2.contourArea(cnt)
                    if area < 2:
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if area >= 40:
                        # Large blob → enemy body; radius scales with size
                        r = int(np.clip(np.sqrt(area) * 0.7, 4, sz // 5))
                        cv2.circle(mask, (cx, cy), r, (255, 0, 0), -1)  # Red
                    else:
                        # Small blob → bullet only if pixels are bright/saturated.
                        # This filters out dull floor decorations sliding due to camera movement.
                        x1b = max(0, cx - 3)
                        y1b = max(0, cy - 3)
                        x2b = min(sz, cx + 4)
                        y2b = min(sz, cy + 4)
                        patch = bullet_color_mask[y1b:y2b, x1b:x2b]
                        if patch.size > 0 and np.any(patch > 0):
                            cv2.circle(mask, (cx, cy), 2, (255, 255, 255), -1)  # White

        self._prev_gray = gray_small.copy()

        # --- Green channel: loot / collectibles ---
        # Roboflow 可用时由模型填充，否则回退到 HSV 颜色检测
        if self.roboflow_detector is None:
            loot_raw = cv2.inRange(hsv, (28, 35, 40), (95, 255, 255))
            loot_contours, _ = cv2.findContours(loot_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in loot_contours:
                area = cv2.contourArea(cnt)
                if area < 3:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                r = int(np.clip(np.sqrt(area) * 0.7, 3, sz // 8))
                cv2.circle(mask, (cx, cy), r, (0, 255, 0), -1)  # Green

        # --- Blue channel: self (detected via the cyan ring under the player sprite) ---
        # The player character always has a distinctive cyan/teal circle beneath it.
        # Detect it by HSV colour and use its centroid as the true player position.
        # Fallback priority: last known position → screen centre.
        # This handles camera-wall-clamp situations where the ring drifts off-centre
        # but is still visible at the screen edge.
        cyan_mask = cv2.inRange(hsv, (85, 80, 80), (100, 255, 255))
        cyan_contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        if cyan_contours:
            largest = max(cyan_contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 3:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    player_x = int(M["m10"] / M["m00"])
                    player_y = int(M["m01"] / M["m00"])
                    self._last_player_pos = (player_x, player_y)
                    detected = True
        if not detected:
            # Use last known position if available, otherwise fall back to centre
            if self._last_player_pos is not None:
                player_x, player_y = self._last_player_pos
            else:
                player_x, player_y = sz // 2, sz // 2
        player_r = max(4, sz // 20)
        cv2.circle(mask, (player_x, player_y), player_r, (0, 0, 255), -1)  # Blue

        return mask

    def _encode_obs(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Encode a frame into obs_channels planes.

        If mask mode is enabled (obs_mask_mode == True):
          Returns a semantic RGB mask: R=enemies, G=loot, B=self, white=bullets.
        If danger mode is enabled (obs_channels == 3):
          Ch0: grayscale (scene structure)
          Ch1: danger channel (bullets / enemies highlighted)
          Ch2: motion channel (frame diff with previous gray)
        Otherwise: single grayscale channel.
        """
        if self.obs_mask_mode:
            return self._build_semantic_mask(frame_rgb)

        sz = self.obs_size

        if not self.obs_danger_enable:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            gray_small = cv2.resize(gray, (sz, sz), interpolation=cv2.INTER_LINEAR)
            return gray_small.astype(np.uint8)

        # Resize frame first to obs_size — all subsequent processing runs on the
        # small image (~36x fewer pixels than full 1280x720 frame).
        frame_small = cv2.resize(frame_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)

        # Danger channel on small frame (much faster)
        danger_small = self._extract_danger_channel(frame_small)

        # Motion channel: frame diff weighted by proximity to center (player position).
        # Threats approaching the player appear brighter; distant motion is dimmed.
        if self._prev_gray is not None and self._prev_gray.shape == gray_small.shape:
            raw_motion = cv2.absdiff(gray_small, self._prev_gray).astype(np.float32)
            if self._proximity_weight is None or self._proximity_weight.shape != raw_motion.shape:
                cy, cx = sz // 2, sz // 2
                ys, xs = np.ogrid[:sz, :sz]
                dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                # Weight: 1.0 at center, 0.35 at corners; exponential falloff
                self._proximity_weight = np.exp(-dist / (sz * 0.45)).astype(np.float32)
            motion = np.clip(raw_motion * (0.35 + 0.65 * self._proximity_weight), 0, 255).astype(np.uint8)
        else:
            motion = np.zeros_like(gray_small)
        self._prev_gray = gray_small.copy()

        # Stack: (H, W, 3)
        return np.stack([gray_small, danger_small, motion], axis=-1).astype(np.uint8)

    def _get_obs(self, frame_rgb: np.ndarray) -> np.ndarray:
        obs_frame = self._encode_obs(frame_rgb)  # (H, W) or (H, W, 3)
        if len(self.obs_frames) == 0:
            for _ in range(self.obs_stack):
                self.obs_frames.append(obs_frame)
        else:
            self.obs_frames.append(obs_frame)
        if self.obs_channels == 1:
            # Stack single-channel frames along last axis: (H, W, stack)
            return np.stack(list(self.obs_frames), axis=-1)
        else:
            # Each frame is (H, W, C), concatenate along channel axis: (H, W, C*stack)
            return np.concatenate(list(self.obs_frames), axis=-1)

    def _obs_diff(self, obs: np.ndarray) -> float:
        # Use the last grayscale channel (first channel of the last frame)
        if self.obs_channels > 1:
            vis = obs[:, :, -self.obs_channels].astype(np.float32)
        else:
            vis = obs[:, :, -1].astype(np.float32)
        if self.prev_obs is None:
            self.prev_obs = vis.copy()
            return 0.0
        diff = float(np.mean(np.abs(vis - self.prev_obs)))
        self.prev_obs = vis.copy()
        return diff

    def _loot_ratio(self, frame_rgb: np.ndarray) -> float:
        if frame_rgb is None or frame_rgb.size == 0:
            return 0.0
        h, w = frame_rgb.shape[:2]
        y1 = int(0.20 * h)
        y2 = int(0.95 * h)
        x1 = int(0.03 * w)
        x2 = int(0.97 * w)
        roi_rgb = frame_rgb[y1:y2, x1:x2]
        if roi_rgb.size == 0:
            return 0.0
        # Resize to small fixed size before HSV processing (much faster)
        roi_rgb = cv2.resize(roi_rgb, (160, 120), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        # Wider green range for different post-process / brightness conditions.
        hsv_mask = cv2.inRange(hsv, (28, 35, 40), (95, 255, 255))
        r = roi_rgb[:, :, 0]
        g = roi_rgb[:, :, 1]
        b = roi_rgb[:, :, 2]
        bgr_mask = ((g > 70) & (g > (r + 18)) & (g > (b + 12))).astype(np.uint8) * 255
        mask = cv2.bitwise_or(hsv_mask, bgr_mask)

        rh, rw = mask.shape[:2]
        cx = int(0.50 * rw)
        cy = int(0.62 * rh)
        rad = max(12, int(0.18 * min(rw, rh)))
        near = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(near, (cx, cy), rad, 255, -1)
        near_ratio = float(np.count_nonzero((mask > 0) & (near > 0))) / float(max(1, np.count_nonzero(near)))
        # Loot reward should reflect near-player pickup events, not global map changes.
        return float(near_ratio)

    def _is_non_battle_state(self, state_name: str) -> bool:
        s = self._normalize_state_name(state_name)
        return s in ("shop", "upgrade", "item_pick", "gameover")

    def _apply_action(self, action: int):
        move_key = None
        if action == 1:
            move_key = "w"
        elif action == 2:
            move_key = "s"
        elif action == 3:
            move_key = "a"
        elif action == 4:
            move_key = "d"
        self.input.set_move_key(move_key)

    def _render_debug(self, frame_rgb: np.ndarray, obs: np.ndarray):
        if not self.debug_mgr.can_render():
            return

        hp = float(self.last_hp)
        x1, y1, x2, y2 = self.cfg.hp_rect
        h, w = frame_rgb.shape[:2]
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
        y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
        hp_crop = frame_rgb[y1:y2, x1:x2]
        hp_panel = cv2.cvtColor(hp_crop, cv2.COLOR_RGB2BGR) if hp_crop.size > 0 else np.zeros((48, 320, 3), dtype=np.uint8)
        cv2.putText(hp_panel, f"hp={hp:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Extract the last frame's grayscale from obs
        if self.obs_mask_mode and self.obs_channels > 1:
            # Mask mode: last frame is semantic RGB (R=enemy, G=loot, B=self)
            # obs is (H, W, 3*stack) — last 3 channels are the newest frame
            r_ch = obs[:, :, -3].astype(np.uint8)
            g_ch = obs[:, :, -2].astype(np.uint8)
            b_ch = obs[:, :, -1].astype(np.uint8)
            # Convert RGB → BGR for cv2 display
            heat = np.stack([b_ch, g_ch, r_ch], axis=-1)
        elif self.obs_channels > 1:
            # obs shape: (H, W, channels*stack), last frame starts at -obs_channels
            gray_ch = obs[:, :, -self.obs_channels].astype(np.float32) / 255.0
            danger_ch = obs[:, :, -self.obs_channels + 1].astype(np.float32) / 255.0
            motion_ch = obs[:, :, -self.obs_channels + 2].astype(np.float32) / 255.0
            # Build RGB heatmap: R=danger, G=safe, B=motion
            heat = np.zeros((gray_ch.shape[0], gray_ch.shape[1], 3), dtype=np.uint8)
            heat[:, :, 2] = np.clip(danger_ch * 255 + motion_ch * 80, 0, 255).astype(np.uint8)  # Red
            heat[:, :, 1] = np.clip((1.0 - danger_ch) * gray_ch * 255, 0, 255).astype(np.uint8)  # Green
            heat[:, :, 0] = np.clip(motion_ch * 200, 0, 255).astype(np.uint8)  # Blue
        else:
            vis = obs[:, :, -1].astype(np.float32) / 255.0
            if self.prev_obs is None or self.prev_obs.shape != vis.shape:
                delta = np.zeros_like(vis)
            else:
                delta = np.abs(vis - (self.prev_obs / 255.0))
            risk = np.clip(0.70 * delta + 0.30 * (1.0 - vis), 0.0, 1.0)
            safe = 1.0 - risk
            heat = np.zeros((vis.shape[0], vis.shape[1], 3), dtype=np.uint8)
            heat[:, :, 1] = (safe * 255).astype(np.uint8)
            heat[:, :, 2] = (risk * 255).astype(np.uint8)
        red_green = cv2.resize(heat, (480, 360), interpolation=cv2.INTER_NEAREST)
        label = f"obs={self.obs_size}x{self.obs_channels}x{self.obs_stack}"
        cv2.putText(red_green, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(red_green, f"state={self.state_name}:{self.state_score:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        shop_panel = np.zeros((460, 860, 3), dtype=np.uint8)
        sdbg = self.shop_policy.debug_snapshot()
        cv2.putText(shop_panel, f"ShopPolicy state={self.state_name}:{self.state_score:.2f} ocr=winmedia", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(shop_panel, f"cycle buys={sdbg['buy_count']} refresh={sdbg['refresh_count']} last={sdbg['action']}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(shop_panel, f"best={float(sdbg['best_score']):.3f} need>={self.cfg.shop_buy_min_score:.2f}", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
        cv2.putText(shop_panel, f"total decision={sdbg['total_decisions']} buy={sdbg['total_buy']} refresh={sdbg['total_refresh']} go={sdbg['total_go']} fail={sdbg['total_buy_fail']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 1)
        cv2.putText(
            shop_panel,
            f"ocr worker={'on' if sdbg.get('ocr_worker_running') else 'off'} submit={sdbg.get('ocr_submit_count', 0)} hit={sdbg.get('ocr_hit_count', 0)} miss={sdbg.get('ocr_miss_count', 0)} age={float(sdbg.get('ocr_last_age_sec', 0.0)):.2f}s",
            (10, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 220, 255),
            1,
        )
        ocr_err = str(sdbg.get("ocr_error", "") or "")
        if ocr_err:
            cv2.putText(shop_panel, f"ocr_error={ocr_err[:84]}", (10, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 180, 255), 1)
            y = 176
        else:
            y = 154
        slots = sdbg.get("slots", [])
        if not slots:
            cv2.putText(shop_panel, "no shop decision yet", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        else:
            for i, info in enumerate(slots[:8]):
                slot = int(info.get("slot", i + 1))
                sc = float(info.get("score", 0.0))
                pcls = float(info.get("primitive_class", 0.0))
                pwp = float(info.get("primitive_weapon", 0.0))
                mbf = float(info.get("melee_buff", 0.0))
                mrg = float(info.get("merge_bonus", 0.0))
                wn = str(info.get("weapon_name", ""))
                color = (0, 220, 0) if (slot - 1) == int(sdbg.get("best_idx", -1)) else (220, 220, 220)
                label = f"S{slot}: score={sc:.3f} prim={pcls:.2f} wpn={pwp:.2f} melee={mbf:.2f}"
                if mrg > 0:
                    label += f" MERGE+{mrg:.1f}"
                if wn:
                    label += f" [{wn}]"
                cv2.putText(shop_panel, label, (10, y + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

        state_panel = np.zeros((460, 860, 3), dtype=np.uint8)
        fps = 0.0 if not self.fps_hist else float(sum(self.fps_hist) / len(self.fps_hist))
        cv2.putText(state_panel, f"State={self.state_name}:{self.state_score:.2f} raw={self.raw_state}:{self.raw_state_score:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 255, 255), 2)
        cv2.putText(state_panel, f"fps={fps:.1f} backend={self.capture_backend} armed={'on' if self.control_armed else 'off'}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1)
        cv2.putText(
            state_panel,
            f"tpl go={self.last_tpl_scores.get('go', 0.0):.2f} choose={self.last_tpl_scores.get('choose', 0.0):.2f} restart={self.last_tpl_scores.get('restart', 0.0):.2f}",
            (12, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 220, 255),
            1,
        )
        rb = self.reward_engine.reward_components_last
        rs = self.reward_engine.episode_component_sums
        cv2.putText(state_panel, f"reward last total={rb.get('total', 0.0):.3f}", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 255, 180), 1)
        cv2.putText(state_panel, f"alive={rb.get('alive_reward', 0.0):.3f} non_battle=-{rb.get('non_battle_penalty', 0.0):.3f}", (12, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        cv2.putText(state_panel, f"damage=-{rb.get('damage_penalty', 0.0):.3f} idle=-{rb.get('idle_penalty', 0.0):.3f} activity=+{rb.get('activity_bonus', 0.0):.3f}", (12, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        cv2.putText(state_panel, f"loot=+{rb.get('loot_collect_bonus', 0.0):.3f} kill=+{rb.get('kill_bonus', 0.0):.3f} death=-{rb.get('death_penalty', 0.0):.3f}", (12, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        cv2.putText(state_panel, f"loot_ev={self.reward_engine.loot_events} kill_ev={self.reward_engine.kill_events} death_ev={self.reward_engine.death_events}", (12, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        cv2.putText(state_panel, f"episode sum={self.episode_reward:.2f} total={self.total_reward:.2f}", (12, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 230, 255), 1)
        cv2.putText(state_panel, f"episode components total={rs.get('total', 0.0):.2f}", (12, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 230, 255), 1)
        cv2.putText(
            state_panel,
            f"anti_stuck same={self.same_action_streak} low={self.low_motion_streak} break_left={self.stuck_break_left} breaks={self.stuck_break_count}",
            (12, 256),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 230, 255),
            1,
        )
        cv2.putText(
            state_panel,
            f"action_hold zero_grace={self.zero_action_streak}/{self.zero_action_grace_frames} last={self.last_battle_action}",
            (12, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 230, 255),
            1,
        )

        self.debug_mgr.show_many(
            {
                DEBUG_WINDOW_HP: hp_panel,
                DEBUG_WINDOW_RED_GREEN: red_green,
                DEBUG_WINDOW_SHOP: shop_panel,
                DEBUG_WINDOW_STATE: state_panel,
            }
        )

    def _check_stop(self):
        self.stop_manager.raise_if_requested()

    def step(self, action):
        self._check_stop()
        self._poll_hotkeys()
        self._render_control_panel()
        self._check_stop()

        now = time.time()
        dt = max(1e-4, now - self.last_step_ts)
        self.last_step_ts = now
        self.fps_hist.append(1.0 / dt)

        frame = self._grab_frame()
        self.raw_state, self.raw_state_score, go_tpl, choose_tpl, restart_tpl = self._infer_state_with_templates(frame)
        self.state_name = self._normalize_state_name(self.raw_state)
        self.state_score = self.raw_state_score
        menu_hp_ratio = float(self._predict_hp(frame))

        self.last_tpl_scores = {
            "go": float(go_tpl),
            "choose": float(choose_tpl),
            "restart": float(restart_tpl),
        }
        self.state_name = self._fallback_menu_state(
            self.state_name,
            state_score=float(self.state_score),
            go_tpl=go_tpl,
            choose_tpl=choose_tpl,
            restart_tpl=restart_tpl,
        )
        tpl_non_battle = max(go_tpl, choose_tpl, restart_tpl)
        if tpl_non_battle >= 0.62:
            self.non_battle_hold_until_ts = max(
                self.non_battle_hold_until_ts,
                time.time() + float(self.cfg.state_non_battle_hold_sec),
            )

        now_ts = time.time()
        # Shop-state hysteresis:
        # lower enter threshold and keep a short hold window to avoid
        # edge flicker where GO button is visible but state toggles.
        hard_non_shop = self.state_name in ("upgrade", "item_pick", "gameover", "restart", "game_over")
        shop_signal = (go_tpl >= 0.48) or (
            self.state_name == "shop" and float(self.state_score) >= min(0.52, float(self.cfg.state_shop_score))
        )
        if hard_non_shop:
            self._shop_soft_until_ts = 0.0
            in_shop = False
        else:
            if shop_signal:
                self._shop_soft_until_ts = now_ts + 0.90
            in_shop = bool(shop_signal or (now_ts < float(self._shop_soft_until_ts)))

        if in_shop and (not self._in_shop_prev):
            self.shop_enter_ts = now_ts
            self.shop_lock_until_ts = now_ts + float(self.cfg.shop_entry_lock_sec)
        elif not in_shop:
            self.shop_lock_until_ts = 0.0
        shop_elapsed_sec = max(0.0, now_ts - float(self.shop_enter_ts)) if in_shop else 0.0
        shop_lock_active = bool(in_shop and now_ts < float(self.shop_lock_until_ts))
        self._in_shop_prev = bool(in_shop)
        script_menu_state = self._script_menu_state(
            self.state_name,
            choose_tpl=choose_tpl,
            restart_tpl=restart_tpl,
        )
        if script_menu_state == "upgrade":
            self._upgrade_menu_streak += 1
        else:
            self._upgrade_menu_streak = 0
            self._upgrade_guard_until_ts = 0.0
            self._upgrade_last_pick_idx = -1
        is_non_battle = self._is_non_battle_state(self.state_name)
        in_non_battle_hold = time.time() < float(self.non_battle_hold_until_ts)
        menu_signal_weak = tpl_non_battle < 0.50
        unknown_but_safe = (self.state_name == "unknown") and menu_signal_weak and (not in_non_battle_hold)
        effective_battle = (self.state_name == "battle") or unknown_but_safe
        if effective_battle:
            self.battle_seen_in_episode = True
        else:
            self._reset_battle_action_trackers()
        gameover_signal = bool(
            script_menu_state == "gameover"
            or self.state_name in ("gameover", "restart", "game_over")
            or restart_tpl >= 0.58
        )

        if time.time() < self.align_until_ts:
            self.input.release_movement()
            obs = self._get_obs(frame)
            self._render_debug(frame, obs)
            return obs, 0.0, False, False, {"phase": "align", "state": self.state_name, "state_score": self.state_score}

        if not self.control_armed:
            self.input.release_movement()
            obs = self._get_obs(frame)
            self._render_debug(frame, obs)
            return obs, 0.0, False, False, {
                "phase": "paused",
                "state": self.state_name,
                "state_score": self.state_score,
                "armed": bool(self.control_armed),
            }

        if self.cfg.shop_policy_enable and (in_shop or go_tpl >= 0.40 or self.state_name == "shop"):
            self.input.release_movement()
            self._focus_game_window_soft()
            dec = self.shop_policy.evaluate(
                frame_rgb=frame,
                in_shop=bool(in_shop),
                state_score=float(self.state_score),
                allow_action=(not shop_lock_active),
            )
            if in_shop:
                obs = self._get_obs(frame)
                self.last_info = {
                    "phase": ("shop_lock" if shop_lock_active else "shop_policy"),
                    "state": self.state_name,
                    "state_score": self.state_score,
                    "hp_ratio": float(menu_hp_ratio),
                    "shop_action": dec.action,
                    "shop_reason": dec.reason,
                    "shop_best_score": dec.best_score,
                    "shop_slot": dec.slot,
                    "shop_lock": bool(shop_lock_active),
                    "shop_elapsed_sec": float(shop_elapsed_sec),
                }
                self._render_debug(frame, obs)
                return obs, 0.0, False, False, dict(self.last_info)

        if bool(script_menu_state) or is_non_battle or in_non_battle_hold:
            self.input.release_movement()
            acted = False
            action_msg = "skip"
            if script_menu_state:
                acted, action_msg = self._menu_interact(script_menu_state)
                action_msg = f"{action_msg}|script_state={script_menu_state}"
            elif in_non_battle_hold:
                action_msg = f"hold_non_battle|go={go_tpl:.2f}|choose={choose_tpl:.2f}|restart={restart_tpl:.2f}"
            obs = self._get_obs(frame)
            if gameover_signal:
                if not acted:
                    acted, action_msg_g = self._menu_interact("gameover")
                    action_msg = f"{action_msg}|{action_msg_g}"
                if not self.battle_seen_in_episode:
                    self.last_info = {
                        "phase": "gameover_wait_no_battle",
                        "state": self.state_name,
                        "state_score": self.state_score,
                        "hp_ratio": float(menu_hp_ratio),
                        "menu_acted": bool(acted),
                        "menu_action": str(action_msg),
                        "menu_hold": bool(in_non_battle_hold),
                        "reward_components_last": dict(self.reward_engine.reward_components_last),
                        "episode_component_sums": dict(self.reward_engine.episode_component_sums),
                        "loot_events": int(self.reward_engine.loot_events),
                        "death_events": int(self.reward_engine.death_events),
                        "death_penalty_applied": bool(self.death_penalty_applied),
                    }
                    self._render_debug(frame, obs)
                    return obs, 0.0, False, False, dict(self.last_info)
                dead_reward = 0.0
                if not self.death_penalty_applied:
                    rb_dead = self.reward_engine.compute(
                        prev_hp=float(self.prev_hp),
                        curr_hp=float(self.prev_hp),
                        is_battle=False,
                        obs_diff=0.0,
                        loot_delta=0.0,
                        dead=True,
                        state_name="gameover",
                        state_elapsed_sec=0.0,
                    )
                    dead_reward = float(rb_dead.total)
                    self.death_penalty_applied = True
                    print(
                        "[reward] death_event "
                        f"cfg_death={self.cfg.reward_death_penalty} "
                        f"applied_death={rb_dead.death_penalty} "
                        f"dead_reward={dead_reward:.3f} "
                        f"episode={self.episode_reward:.3f}"
                    )
                self.episode_reward += dead_reward
                self.total_reward += dead_reward
                self.last_info = {
                    "phase": "gameover_done",
                    "state": self.state_name,
                    "state_score": self.state_score,
                    "hp_ratio": float(menu_hp_ratio),
                    "menu_acted": bool(acted),
                    "menu_action": str(action_msg),
                    "menu_hold": bool(in_non_battle_hold),
                    "reward_components_last": dict(self.reward_engine.reward_components_last),
                    "episode_component_sums": dict(self.reward_engine.episode_component_sums),
                    "loot_events": int(self.reward_engine.loot_events),
                    "death_events": int(self.reward_engine.death_events),
                    "death_penalty_applied": bool(self.death_penalty_applied),
                }
                self._render_debug(frame, obs)
                return obs, dead_reward, True, False, dict(self.last_info)
            self.last_info = {
                "phase": "menu_script",
                "state": self.state_name,
                "state_score": self.state_score,
                "hp_ratio": float(menu_hp_ratio),
                "menu_acted": bool(acted),
                "menu_action": str(action_msg),
                "menu_hold": bool(in_non_battle_hold),
            }
            self._render_debug(frame, obs)
            return obs, 0.0, False, False, dict(self.last_info)

        # Hard gate: PPO movement/action is allowed only in confirmed battle state.
        if not effective_battle:
            self.input.release_movement()
            obs = self._get_obs(frame)
            self.last_info = {
                "phase": "non_battle_wait",
                "state": self.state_name,
                "state_score": self.state_score,
                "hp_ratio": float(menu_hp_ratio),
                "menu_hold": bool(in_non_battle_hold),
                "effective_battle": bool(effective_battle),
                "unknown_but_safe": bool(unknown_but_safe),
                "tpl_non_battle": float(tpl_non_battle),
                "reason": "ppo_locked_outside_battle",
            }
            self._render_debug(frame, obs)
            return obs, 0.0, False, False, dict(self.last_info)

        # battle loop
        total_reward = 0.0
        done = False
        next_frame = frame
        obs = self._get_obs(next_frame)
        repeat_steps = int(max(1, self.cfg.frame_skip))
        for _ in range(repeat_steps):
            self._check_stop()
            planned_action = int(action)
            if planned_action == 0 and self.last_battle_action in (1, 2, 3, 4):
                if self.zero_action_streak < int(self.zero_action_grace_frames):
                    # Smooth one-frame "tap" behavior into a short hold.
                    planned_action = int(self.last_battle_action)
                    self.zero_action_streak += 1
                else:
                    self.zero_action_streak = 0
            else:
                self.zero_action_streak = 0
            applied_action = planned_action
            if self.cfg.anti_stuck_enable and planned_action in (1, 2, 3, 4):
                if self.last_battle_action == planned_action:
                    self.same_action_streak += 1
                else:
                    self.same_action_streak = 1
                motion_stuck = (
                    self.same_action_streak >= int(self.cfg.anti_stuck_same_action_steps)
                    and self.low_motion_streak >= int(self.cfg.anti_stuck_low_motion_steps)
                )
                hard_same_limit = max(8, int(self.cfg.anti_stuck_same_action_steps) * 2)
                hard_stuck = self.same_action_streak >= hard_same_limit
                should_break = self.stuck_break_left > 0 or motion_stuck or hard_stuck
                if should_break:
                    applied_action = self._anti_stuck_pick_alternative(planned_action)
                    if self.stuck_break_left <= 0:
                        break_steps = int(self.cfg.anti_stuck_break_steps)
                        if hard_stuck:
                            break_steps = max(break_steps, 10)
                        self.stuck_break_left = max(0, break_steps - 1)
                        self.stuck_break_count += 1
                    else:
                        self.stuck_break_left -= 1
            else:
                self.same_action_streak = 0
                self.stuck_break_left = 0
            self._apply_action(applied_action)
            if self.cfg.action_sleep_sec > 0:
                time.sleep(float(self.cfg.action_sleep_sec))
            next_frame = self._grab_frame()

            current_hp = self._predict_hp(next_frame)
            obs = self._get_obs(next_frame)
            obs_diff = self._obs_diff(obs)
            if self.cfg.anti_stuck_enable and applied_action in (1, 2, 3, 4):
                if float(obs_diff) < float(self.cfg.anti_stuck_motion_threshold):
                    self.low_motion_streak += 1
                else:
                    self.low_motion_streak = max(0, self.low_motion_streak - 1)
            else:
                self.low_motion_streak = 0
            self.last_battle_action = int(applied_action)

            loot_ratio = self._loot_ratio(next_frame)
            loot_drop = max(0.0, float(self.prev_loot_ratio - loot_ratio))
            loot_spawn_raw = max(0.0, float(loot_ratio - self.prev_loot_ratio))
            is_moving = applied_action in (1, 2, 3, 4)
            if self.loot_reward_cooldown_left > 0:
                self.loot_reward_cooldown_left -= 1
            loot_delta = 0.0
            if (
                is_moving
                and self.loot_reward_cooldown_left <= 0
                and float(self.prev_loot_ratio) >= float(self.loot_min_prev_ratio)
                and float(loot_drop) >= float(self.loot_collect_min_drop)
            ):
                loot_delta = float(loot_drop)
                self.loot_reward_cooldown_left = 8
            # Kill signal: new loot/XP pixels appearing near the player.
            # Throttled by kill_loot_cooldown_left to count one event per kill.
            if self.kill_loot_cooldown_left > 0:
                self.kill_loot_cooldown_left -= 1
            loot_spawn = 0.0
            if self.kill_loot_cooldown_left <= 0 and loot_spawn_raw > 0.0:
                loot_spawn = loot_spawn_raw
                self.kill_loot_cooldown_left = 6
            self.prev_loot_ratio = float(loot_ratio)

            # Death is decided only by explicit gameover/menu signal path.
            dead = False
            self.low_hp_streak = (self.low_hp_streak + 1) if current_hp < 0.02 else 0
            rb = self.reward_engine.compute(
                prev_hp=float(self.prev_hp),
                curr_hp=float(current_hp),
                is_battle=True,
                obs_diff=float(obs_diff),
                loot_delta=float(loot_delta),
                dead=dead,
                state_name="battle",
                state_elapsed_sec=0.0,
                is_moving=bool(is_moving),
                loot_spawn=float(loot_spawn),
            )
            self.prev_hp = float(current_hp)
            self.last_hp = float(current_hp)
            total_reward += float(rb.total)

        self.episode_reward += float(total_reward)
        self.total_reward += float(total_reward)
        self.step_counter += 1
        self.last_info = {
            "phase": ("non_battle" if is_non_battle else "battle"),
            "state": self.state_name,
            "state_score": self.state_score,
            "reward_components_last": dict(self.reward_engine.reward_components_last),
            "episode_component_sums": dict(self.reward_engine.episode_component_sums),
            "loot_events": int(self.reward_engine.loot_events),
            "kill_events": int(self.reward_engine.kill_events),
            "death_events": int(self.reward_engine.death_events),
            "loot_ratio": float(self.prev_loot_ratio),
            "low_hp_streak": int(self.low_hp_streak),
            "same_action_streak": int(self.same_action_streak),
            "low_motion_streak": int(self.low_motion_streak),
            "stuck_break_left": int(self.stuck_break_left),
            "stuck_break_count": int(self.stuck_break_count),
            "loot_reward_cooldown_left": int(self.loot_reward_cooldown_left),
            "loot_collect_min_drop": float(self.loot_collect_min_drop),
            "death_confirm_frames": int(self.cfg.death_confirm_frames),
            "death_penalty_applied": bool(self.death_penalty_applied),
        }

        self._render_debug(next_frame, obs)
        return obs, float(total_reward), bool(done), False, dict(self.last_info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._check_stop()
        self._render_control_panel()

        self.input.release_movement()
        self.shop_policy.reset_episode()   # clears bought_weapons + cycle counters
        self.reward_engine.reset_episode()

        self.obs_frames.clear()
        self.prev_obs = None
        self._prev_gray = None
        self._state_infer_counter = 0
        self.prev_loot_ratio = 0.0
        self.loot_reward_cooldown_left = 0
        self.kill_loot_cooldown_left = 0
        self.low_hp_streak = 0
        self.death_penalty_applied = False
        self.battle_seen_in_episode = False
        self._reset_battle_action_trackers()
        self.step_counter = 0
        self.episode_reward = 0.0
        self.shop_enter_ts = 0.0
        self.shop_lock_until_ts = 0.0
        self._shop_soft_until_ts = 0.0
        self._in_shop_prev = False
        self._upgrade_menu_streak = 0
        self._upgrade_guard_until_ts = 0.0
        self._upgrade_last_pick_idx = -1

        frame = self._grab_frame()
        self.last_hp = self._predict_hp(frame)
        self.prev_hp = self.last_hp

        obs = self._get_obs(frame)
        self._render_debug(frame, obs)
        return obs, {}

    def close(self):
        try:
            self.input.release_movement()
        except Exception:
            pass
        try:
            self.ocr_worker.stop()
        except Exception:
            pass
        try:
            if self.upgrade_ocr is not None:
                self.upgrade_ocr.close()
        except Exception:
            pass
        try:
            self.debug_mgr.close_all()
        except Exception:
            pass
        try:
            cv2.destroyWindow(CONTROL_PANEL_WINDOW)
            cv2.waitKey(1)
        except Exception:
            pass
        try:
            if self.camera is not None:
                self.camera.stop()
        except Exception:
            pass
