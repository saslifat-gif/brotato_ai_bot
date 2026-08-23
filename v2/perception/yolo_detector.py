from pathlib import Path
from typing import Dict

import numpy as np

from .types import Detection, FrameDetections

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YoloDetector:
    """Custom YOLO26 detector with optional persistent tracking."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.25,
        image_size: int = 416,
        device: str = "cpu",
        tracker: str = "bytetrack.yaml",
    ):
        path = Path(weights)
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        if not path.exists():
            raise RuntimeError(
                f"custom detector weights not found: {path}. "
                "Record and label gameplay, then train the v2 detector first."
            )
        self.model = YOLO(str(path))
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = str(device)
        self.tracker = str(tracker)
        self.frame_id = 0

    def detect(self, frame_rgb: np.ndarray, track: bool = True) -> FrameDetections:
        if frame_rgb is None or frame_rgb.size == 0:
            raise ValueError("empty frame")
        self.frame_id += 1
        kwargs = dict(imgsz=self.image_size, conf=self.confidence, device=self.device, verbose=False)
        if track:
            result = self.model.track(frame_rgb, persist=True, tracker=self.tracker, **kwargs)[0]
        else:
            result = self.model.predict(frame_rgb, **kwargs)[0]
        names: Dict[int, str] = dict(getattr(result, "names", {}) or {})
        boxes = getattr(result, "boxes", None)
        detections = []
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            ids = None if boxes.id is None else boxes.id.detach().cpu().numpy().astype(int)
            for i, (box, score, class_id) in enumerate(zip(xyxy, conf, classes)):
                track_id = None if ids is None else int(ids[i])
                detections.append(
                    Detection(
                        label=str(names.get(int(class_id), class_id)).strip().lower(),
                        confidence=float(score),
                        box=tuple(float(v) for v in box),
                        track_id=track_id,
                    )
                )
        h, w = frame_rgb.shape[:2]
        return FrameDetections.from_iterable(w, h, detections, frame_id=self.frame_id)
