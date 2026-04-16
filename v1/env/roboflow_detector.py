"""
Roboflow pretrained Brotato multi-class detector
Model: https://universe.roboflow.com/svamprex/brotato-9qsbr/model/6

Dependencies: pip install inference-sdk
Get API Key: https://app.roboflow.com -> Settings -> API Keys
Or set environment variable: set ROBOFLOW_API_KEY=your_key_here

Detection runs asynchronously in a background thread, without blocking the game main loop.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, NamedTuple, Optional, Tuple

import cv2
import numpy as np

ROBOFLOW_MODEL_ID = "brotato-9qsbr/6"

# ==================== Class Configuration ====================
# Observation map channel mapping:
#   Red channel   = enemies (danger)
#   Green channel = loot (reward)
#   White dots    = bullets/projectiles (dangerous but small)
#   Blue channel  = player (fixed, detected via cyan ring)

ENEMY_CLASSES = {
    "enemy",
}

BULLET_CLASSES = {
    "long_proj",
}

LOOT_CLASSES: set[str] = set()

PLAYER_CLASSES = {
    "you",
}

STRUCTURE_CLASSES = {
    "structure",
}
# =================================================

_EMPTY_CHANNELS = None  # lazy initialization


def _make_empty(out_size: int) -> Dict[str, object]:
    z = np.zeros((out_size, out_size), dtype=np.uint8)
    return {
        "enemy": z, "loot": z, "bullet": z,
        "player": z, "structure": z, "player_pos": None,
    }


class Detection(NamedTuple):
    class_name: str
    confidence: float
    cx: float
    cy: float
    w: float
    h: float


class RoboflowMonsterDetector:
    """
    Wraps the Roboflow pretrained Brotato detection model.
    Detection runs asynchronously in a background thread; build_channels() returns
    the previous frame's result immediately, avoiding cloud API latency blocking the game main loop.
    """

    def __init__(
        self,
        api_key: str = "",
        confidence: float = 0.35,
    ):
        if not api_key:
            api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Roboflow API Key is required.\n"
                "Get it at: https://app.roboflow.com -> Settings -> API Keys\n"
                "Or set environment variable: set ROBOFLOW_API_KEY=xxx"
            )

        self.confidence = confidence
        self._seen_classes: set[str] = set()

        from inference_sdk import InferenceHTTPClient
        self._client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key,
        )
        print(f"[RoboflowDetector] Initialized successfully, model: {ROBOFLOW_MODEL_ID}")

        # Background async detection
        self._lock = threading.Lock()
        self._cached: Optional[Dict] = None   # latest detection result
        self._pending: Optional[Tuple] = None  # (frame_bgr, out_size)
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Background detection thread
    # ------------------------------------------------------------------

    def _loop(self):
        while True:
            self._event.wait()
            self._event.clear()
            job = self._pending
            if job is None:
                continue
            frame_bgr, out_size = job
            try:
                channels = self._detect_sync(frame_bgr, out_size)
                with self._lock:
                    self._cached = channels
            except Exception as e:
                print(f"[RoboflowDetector] Detection failed: {e}")

    def _detect_sync(self, frame_bgr: np.ndarray, out_size: int) -> Dict:
        """Calls the API and generates per-channel masks (runs in the background thread)."""
        h, w = frame_bgr.shape[:2]

        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        raw = self._client.infer(buf.tobytes(), model_id=ROBOFLOW_MODEL_ID)

        enemy_mask     = np.zeros((out_size, out_size), dtype=np.uint8)
        loot_mask      = np.zeros((out_size, out_size), dtype=np.uint8)
        bullet_mask    = np.zeros((out_size, out_size), dtype=np.uint8)
        player_mask    = np.zeros((out_size, out_size), dtype=np.uint8)
        structure_mask = np.zeros((out_size, out_size), dtype=np.uint8)
        player_pos     = None

        for p in raw.get("predictions", []):
            if p["confidence"] < self.confidence:
                continue
            name = p["class"].lower().strip()
            self._seen_classes.add(name)

            cx_n = p["x"] / w
            cy_n = p["y"] / h
            w_n  = p["width"] / w
            h_n  = p["height"] / h
            px = int(np.clip(cx_n * out_size, 0, out_size - 1))
            py = int(np.clip(cy_n * out_size, 0, out_size - 1))
            area = w_n * h_n * out_size * out_size

            if name in ENEMY_CLASSES:
                r = int(np.clip(np.sqrt(area) * 0.45, 3, out_size // 6))
                cv2.circle(enemy_mask, (px, py), r, 255, -1)
            elif name in BULLET_CLASSES:
                r = int(np.clip(np.sqrt(area) * 0.35, 2, out_size // 10))
                cv2.circle(bullet_mask, (px, py), r, 255, -1)
            elif name in LOOT_CLASSES:
                r = int(np.clip(np.sqrt(area) * 0.40, 2, out_size // 8))
                cv2.circle(loot_mask, (px, py), r, 255, -1)
            elif name in PLAYER_CLASSES:
                r = max(4, out_size // 20)
                cv2.circle(player_mask, (px, py), r, 255, -1)
                player_pos = (px, py)
            elif name in STRUCTURE_CLASSES:
                x1 = int(np.clip((cx_n - w_n / 2) * out_size, 0, out_size - 1))
                y1 = int(np.clip((cy_n - h_n / 2) * out_size, 0, out_size - 1))
                x2 = int(np.clip((cx_n + w_n / 2) * out_size, 0, out_size - 1))
                y2 = int(np.clip((cy_n + h_n / 2) * out_size, 0, out_size - 1))
                cv2.rectangle(structure_mask, (x1, y1), (x2, y2), 80, 1)

        return {
            "enemy": enemy_mask, "loot": loot_mask, "bullet": bullet_mask,
            "player": player_mask, "structure": structure_mask,
            "player_pos": player_pos,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_channels(self, frame_bgr: np.ndarray, out_size: int) -> Dict:
        """
        Submits the current frame for background detection and immediately returns
        the previous frame's result. The first frame blocks once; all subsequent
        calls are async with no delay.
        """
        # Submit new frame
        self._pending = (frame_bgr.copy(), out_size)
        self._event.set()

        with self._lock:
            if self._cached is not None:
                return self._cached

        # First frame: wait synchronously for a result
        self._event.wait()
        with self._lock:
            if self._cached is not None:
                return self._cached
        return _make_empty(out_size)

    def print_classes(self, frame_bgr: Optional[np.ndarray] = None) -> None:
        """Prints all detected class names, useful for verifying the configuration is correct."""
        if frame_bgr is not None:
            self._detect_sync(frame_bgr, 160)

        if not self._seen_classes:
            print("[RoboflowDetector] No images detected yet — pass a frame first")
            return

        print("\n[RoboflowDetector] Detected class names:")
        for name in sorted(self._seen_classes):
            if name in ENEMY_CLASSES:
                tag = " <- red channel [enemy]"
            elif name in BULLET_CLASSES:
                tag = " <- white dot [bullet]"
            elif name in LOOT_CLASSES:
                tag = " <- green channel [loot]"
            elif name in PLAYER_CLASSES:
                tag = " <- blue channel [player]"
            elif name in STRUCTURE_CLASSES:
                tag = " <- blue channel [structure outline]"
            else:
                tag = " <- ⚠ unclassified"
            print(f"  '{name}'{tag}")

    def enemy_channel(self, frame_bgr: np.ndarray, out_size: int) -> np.ndarray:
        return self.build_channels(frame_bgr, out_size)["enemy"]
