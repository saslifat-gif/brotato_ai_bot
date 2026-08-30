"""Compatibility imports for the active bridge transport."""

from brotato_ai.bridge.client import BridgeClient, BridgeDisconnected

BridgeServer = BridgeClient

__all__ = ["BridgeClient", "BridgeDisconnected", "BridgeServer"]
