"""Versioned JSONL record contracts."""

from __future__ import annotations

import time
from typing import Any, Mapping

from brotato_ai.domain.state import StateSnapshot


RAW_RECORD_SCHEMA_VERSION = 2


class RecordSchemaError(ValueError):
    pass


def normalize_raw_record(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable raw-state record without silently changing old fields."""

    if not isinstance(message, Mapping):
        raise RecordSchemaError("raw record must be a mapping")
    if message.get("type") not in {"raw_state", "state"}:
        raise RecordSchemaError(f"unsupported raw record type: {message.get('type')!r}")
    snapshot = StateSnapshot.from_payload(message)
    record = snapshot.to_dict()
    record["type"] = "raw_state"
    record["record_type"] = "state_snapshot"
    record["schema_version"] = RAW_RECORD_SCHEMA_VERSION
    record.setdefault("recorded_at_ms", int(time.time() * 1000.0))
    action = record.get("action", -1)
    try:
        action = int(action)
    except (TypeError, ValueError):
        action = -1
    record["action"] = action if 0 <= action < 9 else -1
    return record


def validate_schema_version(record: Mapping[str, Any]) -> int:
    """Accept legacy v1/unversioned data explicitly; reject future meanings."""

    raw = record.get("schema_version", 1)
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise RecordSchemaError(f"invalid schema_version: {raw!r}") from exc
    if version not in {1, RAW_RECORD_SCHEMA_VERSION}:
        raise RecordSchemaError(
            f"unsupported schema_version={version}; supported=1,{RAW_RECORD_SCHEMA_VERSION}"
        )
    return version

