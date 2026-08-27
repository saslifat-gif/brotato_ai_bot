"""Deterministic immutable JSONL replay loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from brotato_ai.data.schema import validate_schema_version
from brotato_ai.domain.state import StateSnapshot


class JsonlReplay:
    def __init__(self, path: Path, *, max_records: int = 0, stride: int = 1):
        self.path = Path(path)
        self.max_records = max(0, int(max_records))
        self.stride = max(1, int(stride))

    def records(self) -> Iterator[tuple[StateSnapshot, int]]:
        emitted = 0
        accepted = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    raw = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(raw, dict) or raw.get("type") not in {
                    "raw_state",
                    "state",
                }:
                    continue
                validate_schema_version(raw)
                try:
                    action = int(raw.get("action", -1))
                except (TypeError, ValueError):
                    continue
                if not 0 <= action < 9:
                    continue
                if accepted % self.stride:
                    accepted += 1
                    continue
                accepted += 1
                yield StateSnapshot.from_payload(raw), action
                emitted += 1
                if self.max_records and emitted >= self.max_records:
                    break

