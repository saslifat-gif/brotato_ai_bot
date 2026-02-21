import ctypes
import signal
import threading
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class TrainingStopRequested(RuntimeError):
    """Raised when a cooperative global stop is requested."""


class StopManager:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    def request_stop(self, reason: str) -> bool:
        with self._lock:
            first = not self._event.is_set()
            if first:
                self._reason = str(reason or "unknown")
                self._event.set()
            return first

    def should_stop(self) -> bool:
        return self._event.is_set()

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_requested(self):
        if self.should_stop():
            raise TrainingStopRequested(self.reason() or "stop_requested")


class StopTrainingCallback(BaseCallback):
    """Stops SB3 learn loop as soon as StopManager is set."""

    def __init__(self, stop_manager: StopManager, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.stop_manager = stop_manager

    def _on_step(self) -> bool:
        return not self.stop_manager.should_stop()


_CONSOLE_HANDLER_REF = None


def install_stop_handlers(stop_manager: StopManager):
    def _signal_handler(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        if stop_manager.request_stop(sig_name):
            print(f"[control] stop requested ({sig_name})")

    handlers = [signal.SIGINT]
    if hasattr(signal, "SIGBREAK"):
        handlers.append(signal.SIGBREAK)
    for sig in handlers:
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    # Register Win32 console control handler for robust Ctrl+C handling.
    if hasattr(ctypes, "windll"):
        kernel32 = ctypes.windll.kernel32
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        def _console_handler(ctrl_type: int):
            # 0: CTRL_C_EVENT, 1: CTRL_BREAK_EVENT, 2: CTRL_CLOSE_EVENT
            mapping = {
                0: "CTRL_C_EVENT",
                1: "CTRL_BREAK_EVENT",
                2: "CTRL_CLOSE_EVENT",
                5: "CTRL_LOGOFF_EVENT",
                6: "CTRL_SHUTDOWN_EVENT",
            }
            reason = mapping.get(int(ctrl_type), f"CTRL_{ctrl_type}")
            if stop_manager.request_stop(reason):
                print(f"[control] stop requested ({reason})")
            return True

        global _CONSOLE_HANDLER_REF
        try:
            _CONSOLE_HANDLER_REF = handler_type(_console_handler)
            kernel32.SetConsoleCtrlHandler(_CONSOLE_HANDLER_REF, True)
        except Exception:
            _CONSOLE_HANDLER_REF = None
