from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Box
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


@dataclass(frozen=True)
class FrameDetections:
    width: int
    height: int
    items: Tuple[Detection, ...] = field(default_factory=tuple)
    frame_id: int = 0

    @classmethod
    def from_iterable(
        cls,
        width: int,
        height: int,
        items: Iterable[Detection],
        frame_id: int = 0,
    ) -> "FrameDetections":
        return cls(max(1, int(width)), max(1, int(height)), tuple(items), int(frame_id))

    def by_label(self, label: str) -> Tuple[Detection, ...]:
        wanted = str(label).strip().lower()
        return tuple(item for item in self.items if item.label.strip().lower() == wanted)

    def best(self, label: str) -> Optional[Detection]:
        matches = self.by_label(label)
        return max(matches, key=lambda item: item.confidence) if matches else None

