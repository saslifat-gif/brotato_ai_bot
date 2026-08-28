"""Versioned JSON-lines protocol shared by Python and the Godot mod."""

from __future__ import annotations

import json
from typing import Any, Mapping

from brotato_ai.domain.actions import MoveAction


PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4242
MAX_STATE_HZ = 60.0


class BridgeProtocolError(RuntimeError):
    pass


def encode_message(message: Mapping[str, Any]) -> bytes:
    payload = dict(message)
    payload.setdefault("protocol", PROTOCOL_VERSION)
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def decode_message(line: bytes | str) -> dict[str, Any]:
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
        message = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeProtocolError(f"invalid bridge JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise BridgeProtocolError("bridge message must be a JSON object")
    version = message.get("protocol")
    if version != PROTOCOL_VERSION:
        raise BridgeProtocolError(
            f"protocol mismatch: game={version!r} trainer={PROTOCOL_VERSION}"
        )
    message_type = message.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise BridgeProtocolError("bridge message is missing a string 'type'")
    return message


def action_message(action: int, sequence: int) -> dict[str, Any]:
    try:
        normalized = MoveAction(int(action))
    except (TypeError, ValueError) as exc:
        raise BridgeProtocolError(f"invalid movement action: {action!r}") from exc
    return {"type": "action", "sequence": int(sequence), "action": int(normalized)}


def reset_message(sequence: int) -> dict[str, Any]:
    return {"type": "reset", "sequence": int(sequence)}


def ui_action_message(target: str, sequence: int) -> dict[str, Any]:
    normalized = str(target).strip()
    if not normalized.startswith("/") or len(normalized) > 1024:
        raise BridgeProtocolError(f"invalid UI action target: {target!r}")
    return {"type": "ui_action", "sequence": int(sequence), "target": normalized}


def configure_message(*, state_hz: float) -> dict[str, Any]:
    normalized = float(state_hz)
    if not 4.0 <= normalized <= MAX_STATE_HZ:
        raise BridgeProtocolError(
            f"state_hz must be between 4 and {MAX_STATE_HZ:g}: {state_hz!r}"
        )
    return {"type": "configure", "state_hz": normalized}


def training_pause_message(paused: bool) -> dict[str, Any]:
    return {"type": "training_pause", "paused": bool(paused)}
