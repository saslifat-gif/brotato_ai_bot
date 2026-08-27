"""Local TCP bridge with movement writes reserved for the final arbiter."""

from __future__ import annotations

import socket
import time
from typing import Any

from brotato_ai.bridge.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    BridgeProtocolError,
    action_message,
    decode_message,
    encode_message,
)


class BridgeDisconnected(RuntimeError):
    pass


class BridgeClient:
    """Own the game transport while preventing ad-hoc action messages.

    Configuration, reset, pause, and UI messages use :meth:`send`. Movement
    messages can only pass through the package-private method used by
    ``FinalActionWriter``.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = str(host)
        self.port = int(port)
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None
        self._buffer = bytearray()
        self._connection_generation = 0
        self._last_action: dict[str, Any] | None = None
        self.last_hello: dict[str, Any] | None = None
        # Process-local transport counters; these are intentionally not reset
        # between episodes so TensorBoard exposes bridge health for the run.
        self.received_state_count = 0
        self.stale_state_count = 0
        self.dropped_state_count = 0
        self.last_tick_gap = 0
        self._last_received_tick = -1

    @property
    def connected(self) -> bool:
        return self._client is not None

    def start(self) -> None:
        if self._listener is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(1)
        self._listener = listener
        print(f"[bridge] listening on {self.host}:{self.port}")

    def _drop_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
        self._client = None
        self._buffer.clear()
        self.last_hello = None

    def _accept(self, deadline: float) -> None:
        self.start()
        assert self._listener is not None
        while self._client is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for the Brotato bridge mod")
            self._listener.settimeout(min(1.0, remaining))
            try:
                client, address = self._listener.accept()
            except socket.timeout:
                continue
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                if self._last_action is not None:
                    client.sendall(encode_message(self._last_action))
            except OSError:
                client.close()
                continue
            self._client = client
            self._connection_generation += 1
            self._buffer.clear()
            print(f"[bridge] game connected from {address[0]}:{address[1]}")

    def _send_unchecked(self, message: dict[str, Any], timeout_sec: float) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        if self._client is None:
            self._accept(deadline)
        assert self._client is not None
        try:
            self._client.sendall(encode_message(message))
        except OSError as exc:
            self._drop_client()
            raise BridgeDisconnected(f"game connection lost while sending: {exc}") from exc

    def send(self, message: dict[str, Any], timeout_sec: float = 10.0) -> None:
        """Send a non-movement message.

        Rejecting ``type=action`` makes duplicate action writers fail loudly.
        """

        message_type = message.get("type")
        if message_type == "action":
            raise RuntimeError("movement actions must be sent by FinalActionWriter")
        if message_type == "reset":
            self._last_action = None
        self._send_unchecked(message, timeout_sec)

    def _write_final_action(
        self, action: int, sequence: int, timeout_sec: float = 10.0
    ) -> None:
        message = action_message(action, sequence)
        self._last_action = dict(message)
        self._send_unchecked(message, timeout_sec)

    def receive(self, timeout_sec: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while True:
            separator = self._buffer.find(b"\n")
            if separator >= 0:
                line = bytes(self._buffer[:separator])
                del self._buffer[: separator + 1]
                if not line.strip():
                    continue
                return decode_message(line)
            if self._client is None:
                self._accept(deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a bridge message")
            assert self._client is not None
            self._client.settimeout(min(1.0, remaining))
            try:
                chunk = self._client.recv(65536)
            except socket.timeout:
                continue
            except OSError as exc:
                self._drop_client()
                raise BridgeDisconnected(f"game connection lost while receiving: {exc}") from exc
            if not chunk:
                self._drop_client()
                raise BridgeDisconnected("game closed the bridge connection")
            self._buffer.extend(chunk)
            if len(self._buffer) > 4 * 1024 * 1024:
                self._drop_client()
                raise BridgeProtocolError("bridge message exceeded 4 MiB")

    def wait_for_state(
        self,
        timeout_sec: float = 300.0,
        after_tick: int | None = None,
        minimum_sequence: int | None = None,
        combat_only: bool = False,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        connection_generation = self._connection_generation
        minimum_tick = after_tick
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for game state")
            try:
                message = self.receive(remaining)
            except BridgeDisconnected:
                continue
            if self._connection_generation != connection_generation:
                connection_generation = self._connection_generation
                minimum_tick = None
                self._last_received_tick = -1
            if message["type"] == "hello":
                self.last_hello = message
                print(
                    "[bridge] handshake "
                    f"mod={message.get('mod_version', '?')} game={message.get('game_version', '?')}"
                )
                continue
            if message["type"] != "state":
                continue
            tick = int(message.get("tick", -1))
            if minimum_tick is not None and tick <= int(minimum_tick):
                self.stale_state_count += 1
                self.dropped_state_count += 1
                continue
            if minimum_sequence is not None and int(message.get("sequence", -1)) < int(
                minimum_sequence
            ):
                self.stale_state_count += 1
                self.dropped_state_count += 1
                continue
            if self.received_state_count:
                self.last_tick_gap = max(0, tick - int(self._last_received_tick))
                if self.last_tick_gap > 1:
                    self.dropped_state_count += self.last_tick_gap - 1
            self._last_received_tick = tick
            self.received_state_count += 1
            if combat_only and message.get("phase") != "combat":
                continue
            return message

    def close(self) -> None:
        self._drop_client()
        self._last_action = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        self._listener = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

