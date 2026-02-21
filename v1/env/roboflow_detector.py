"""
Roboflow 预训练 Brotato 多类别检测器
模型：https://universe.roboflow.com/svamprex/brotato-9qsbr/model/6

依赖：pip install inference-sdk
获取 API Key：https://app.roboflow.com -> Settings -> API Keys
或设置环境变量：set ROBOFLOW_API_KEY=your_key_here

检测在后台线程异步执行，不阻塞游戏主循环。
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

ROBOFLOW_MODEL_ID = "brotato-9qsbr/6"

# ==================== 类别配置 ====================
# 观测图通道映射：
#   红通道 = 怪物（危险）
#   绿通道 = 掉落物（收益）
#   白色点 = 子弹/飞射物（危险但小）
#   蓝通道 = 玩家（固定，由青色环检测）

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

_EMPTY_CHANNELS = None  # 延迟初始化


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
    封装 Roboflow 预训练 Brotato 检测模型。
    检测在后台线程异步执行，build_channels() 立即返回上一帧的结果，
    避免云端 API 延迟阻塞游戏主循环。
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
                "需要 Roboflow API Key。\n"
                "获取地址：https://app.roboflow.com -> Settings -> API Keys\n"
                "或设置环境变量：set ROBOFLOW_API_KEY=xxx"
            )

        self.confidence = confidence
        self._seen_classes: set[str] = set()

        from inference_sdk import InferenceHTTPClient
        self._client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key,
        )
        print(f"[RoboflowDetector] 初始化成功，模型: {ROBOFLOW_MODEL_ID}")

        # 后台异步检测
        self._lock = threading.Lock()
        self._cached: Optional[Dict] = None   # 最新检测结果
        self._pending: Optional[Tuple] = None  # (frame_bgr, out_size)
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 后台检测线程
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
                print(f"[RoboflowDetector] 检测失败: {e}")

    def _detect_sync(self, frame_bgr: np.ndarray, out_size: int) -> Dict:
        """实际调用 API 并生成各通道掩码（在后台线程中执行）。"""
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
    # 公共接口
    # ------------------------------------------------------------------

    def build_channels(self, frame_bgr: np.ndarray, out_size: int) -> Dict:
        """
        提交当前帧到后台检测，立即返回上一帧的检测结果。
        第一帧会同步等待（阻塞一次），之后全部异步无延迟。
        """
        # 提交新帧
        self._pending = (frame_bgr.copy(), out_size)
        self._event.set()

        with self._lock:
            if self._cached is not None:
                return self._cached

        # 首帧：同步等待有结果
        self._event.wait()
        with self._lock:
            if self._cached is not None:
                return self._cached
        return _make_empty(out_size)

    def print_classes(self, frame_bgr: Optional[np.ndarray] = None) -> None:
        """打印检测到的所有类别名称，用于确认配置是否正确。"""
        if frame_bgr is not None:
            self._detect_sync(frame_bgr, 160)

        if not self._seen_classes:
            print("[RoboflowDetector] 尚未检测任何图像，请先传入一帧图像")
            return

        print("\n[RoboflowDetector] 检测到的类别名称：")
        for name in sorted(self._seen_classes):
            if name in ENEMY_CLASSES:
                tag = " <- 红通道【怪物】"
            elif name in BULLET_CLASSES:
                tag = " <- 白色点【子弹】"
            elif name in LOOT_CLASSES:
                tag = " <- 绿通道【掉落物】"
            elif name in PLAYER_CLASSES:
                tag = " <- 蓝通道【玩家】"
            elif name in STRUCTURE_CLASSES:
                tag = " <- 蓝通道【建筑轮廓】"
            else:
                tag = " <- ⚠ 未分类"
            print(f"  '{name}'{tag}")

    def enemy_channel(self, frame_bgr: np.ndarray, out_size: int) -> np.ndarray:
        return self.build_channels(frame_bgr, out_size)["enemy"]
