import time
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v1.runtime.capture import create_camera
from v1.runtime.input_driver import InputDriver
from v2.config import V2Config
from v2.perception.vectorizer import CombatStateVectorizer
from v2.perception.yolo_detector import YoloDetector
from v2.runtime.ui_controller import UiAction, UiController
from v2.runtime.window import client_screen_rect, find_game_window, monitor_for_region


class BrotatoVectorEnv(gym.Env):
    """Detector-driven Brotato environment with a compact vector observation."""

    metadata = {"render_modes": []}
    _ACTION_KEYS = {
        0: (),
        1: ("w",),
        2: ("s",),
        3: ("a",),
        4: ("d",),
        5: ("w", "a"),
        6: ("w", "d"),
        7: ("s", "a"),
        8: ("s", "d"),
    }

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        self.hwnd = find_game_window(cfg.window_title)
        self.region = client_screen_rect(self.hwnd)
        monitor_index, monitor_origin = monitor_for_region(self.region)
        self.camera = create_camera(
            cfg.capture_backend,
            self.region,
            monitor_index=monitor_index,
            monitor_origin=monitor_origin,
            target_fps=60,
        )
        self.input = InputDriver(self.hwnd, input_mode="physical_foreground", move_physical=True)
        self.combat_detector = YoloDetector(
            str(cfg.combat_weights),
            confidence=cfg.detector_confidence,
            image_size=cfg.detector_image_size,
            device=cfg.detector_device,
            tracker="bytetrack.yaml",
        )
        self.ui_detector: Optional[YoloDetector] = None
        if cfg.ui_weights.exists():
            self.ui_detector = YoloDetector(
                str(cfg.ui_weights),
                confidence=max(0.50, cfg.detector_confidence),
                image_size=cfg.detector_image_size,
                device=cfg.detector_device,
                tracker="bytetrack.yaml",
            )
        self.ui = UiController(minimum_confidence=0.65)
        self.vectorizer = CombatStateVectorizer()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.vectorizer.observation_size,),
            dtype=np.float32,
        )
        self.previous_action = 0
        self.previous_hp = 1.0
        self.missing_player_frames = 0
        self.last_frame = None
        self.last_ui_click_ts = 0.0
        self.last_ui_action = UiAction.WAIT

    def _frame(self, timeout_sec: float = 1.0) -> np.ndarray:
        deadline = time.time() + max(0.05, float(timeout_sec))
        while time.time() < deadline:
            frame = self.camera.get_latest_frame()
            if frame is not None and frame.size > 0:
                self.last_frame = frame
                return frame
            time.sleep(0.01)
        if self.last_frame is not None:
            return self.last_frame.copy()
        left, top, right, bottom = self.region
        return np.zeros((max(1, bottom - top), max(1, right - left), 3), dtype=np.uint8)

    @staticmethod
    def _hp_ratio(frame_rgb: np.ndarray) -> float:
        h, w = frame_rgb.shape[:2]
        # Reference HP bar: (23,22)-(342,70) at 1920x1080.
        x1, y1 = int(w * 23 / 1920), int(h * 22 / 1080)
        x2, y2 = int(w * 342 / 1920), int(h * 70 / 1080)
        crop = frame_rgb[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        if crop.size == 0:
            return 1.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 90, 90), (12, 255, 255)),
            cv2.inRange(hsv, (170, 90, 90), (180, 255, 255)),
        )
        columns = np.max(red, axis=0)
        return float(np.clip(np.count_nonzero(columns) / max(1, columns.shape[0]), 0.0, 1.0))

    def _handle_ui(self, frame: np.ndarray):
        if self.ui_detector is None:
            return None
        detections = self.ui_detector.detect(frame, track=False)
        decision = self.ui.decide(detections)
        if decision.action == UiAction.WAIT or decision.client_point is None:
            return decision
        now = time.time()
        if decision.action == self.last_ui_action and now - self.last_ui_click_ts < 0.75:
            return type(decision)(UiAction.WAIT, reason="ui_click_cooldown")
        self.input.release_movement()
        result = self.input.click_client_point(decision.client_point)
        if result.ok:
            self.last_ui_click_ts = now
            self.last_ui_action = decision.action
        print(
            f"[v2-ui] action={decision.action.value} confidence={decision.confidence:.2f} "
            f"point={decision.client_point} click={result.ok}:{result.error or '-'}"
        )
        return decision

    def step(self, action):
        action = int(action)
        self.input.set_move_keys(self._ACTION_KEYS.get(action, ()))
        time.sleep(self.cfg.action_interval_sec)
        frame = self._frame()

        ui_decision = self._handle_ui(frame)
        if ui_decision is not None and ui_decision.action != UiAction.WAIT:
            terminated = ui_decision.action == UiAction.RESTART
            reward = -25.0 if terminated else (5.0 if ui_decision.action == UiAction.NEXT_WAVE else 0.0)
            combat = self.combat_detector.detect(frame)
            obs = self.vectorizer.build(combat, self.previous_hp, previous_action=action)
            self.previous_action = action
            return obs, reward, terminated, False, {
                "phase": "ui",
                "ui_action": ui_decision.action.value,
                "ui_confidence": ui_decision.confidence,
            }

        detections = self.combat_detector.detect(frame)
        player_visible = detections.best("player") is not None
        self.missing_player_frames = 0 if player_visible else self.missing_player_frames + 1
        hp = self._hp_ratio(frame)
        damage = max(0.0, self.previous_hp - hp)
        reward = 0.01 - 8.0 * damage
        if action == 0:
            reward -= 0.01
        # If the combat detector loses the player for a sustained period,
        # wait for the UI detector instead of clicking or declaring death.
        truncated = self.missing_player_frames >= 150 and self.ui_detector is None
        obs = self.vectorizer.build(detections, hp_ratio=hp, previous_action=action)
        self.previous_hp = hp
        self.previous_action = action
        return obs, float(reward), False, bool(truncated), {
            "phase": "battle" if player_visible else "uncertain",
            "hp_ratio": hp,
            "player_visible": player_visible,
            "detections": len(detections.items),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.input.release_movement()
        self.vectorizer.reset()
        self.previous_action = 0
        self.missing_player_frames = 0
        self.last_ui_action = UiAction.WAIT
        self.last_ui_click_ts = 0.0
        frame = self._frame()
        self.previous_hp = self._hp_ratio(frame)
        detections = self.combat_detector.detect(frame)
        obs = self.vectorizer.build(detections, hp_ratio=self.previous_hp)
        return obs, {"phase": "reset"}

    def close(self):
        try:
            self.input.release_movement()
        finally:
            self.camera.stop()
