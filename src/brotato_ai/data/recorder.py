"""Independent high-rate recorder with asynchronous disk writes."""

from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
from pathlib import Path

from brotato_ai.data.cache import BoundedCache
from brotato_ai.data.schema import normalize_raw_record
from brotato_ai.domain.decisions import DecisionTrace
from brotato_ai.performance import RuntimeProfiler


class DecisionTraceLogger:
    """Append decision traces asynchronously without blocking control.

    The queue is bounded.  If disk falls behind, the oldest pending trace is
    dropped and ``dropped_count`` is incremented; the control loop never waits
    for file I/O.  The worker serializes traces so JSON formatting is outside
    the control thread as well.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        queue_size: int = 4096,
        profiler: RuntimeProfiler | None = None,
    ):
        self.path = Path(path) if path is not None else None
        self.queue: queue.Queue[DecisionTrace | None] = queue.Queue(
            maxsize=max(16, int(queue_size))
        )
        self.profiler = profiler or RuntimeProfiler.disabled()
        self.thread: threading.Thread | None = None
        self._closed = False
        self._lock = threading.Lock()
        self.enqueued_count = 0
        self.written_count = 0
        self.dropped_count = 0

    def _start(self) -> None:
        if self.path is None or self.thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run, name="decision-trace-writer", daemon=True
        )
        self.thread.start()

    def record(self, trace: DecisionTrace) -> None:
        if self.path is None:
            return
        with self._lock:
            if self._closed:
                return
            self._start()
            try:
                self.queue.put_nowait(trace)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(trace)
                except queue.Full:
                    self.dropped_count += 1
                    self.profiler.count("recording_dropped")
                    return
                self.dropped_count += 1
                self.profiler.count("recording_dropped")
            self.enqueued_count += 1
            self.profiler.count("recording_enqueued")
            self.profiler.value("recording_queue_depth", self.queue.qsize())

    def _run(self) -> None:
        assert self.path is not None
        try:
            with self.path.open("a", encoding="utf-8", buffering=1024 * 1024) as handle:
                while True:
                    trace = self.queue.get()
                    try:
                        if trace is None:
                            break
                        started = self.profiler.begin("recording_worker_serialize_and_io")
                        handle.write(
                            json.dumps(
                                trace.to_dict(), separators=(",", ":"), allow_nan=False
                            )
                            + "\n"
                        )
                        self.profiler.end("recording_worker_serialize_and_io", started)
                        self.written_count += 1
                    finally:
                        self.queue.task_done()
                handle.flush()
        except BaseException as exc:
            self.profiler.count("recording_worker_errors")
            self.profiler.value("recording_worker_error", 1.0)
            # Keep the control loop alive; the final report exposes the error
            # counter and the written/enqueued counts show what was retained.
            _ = exc

    def close(self) -> None:
        with self._lock:
            if self.thread is None or self._closed:
                return
            self._closed = True
            while True:
                try:
                    self.queue.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.queue.task_done()
                    except queue.Empty:
                        break
            thread = self.thread
        thread.join()

    @property
    def backlog(self) -> int:
        return self.queue.qsize()


class AsyncJsonlWriter:
    def __init__(self, path: Path, *, queue_size: int = 4096):
        self.path = Path(path)
        self.queue: queue.Queue[dict | None] = queue.Queue(maxsize=max(16, queue_size))
        self.error: BaseException | None = None
        self.dropped_count = 0
        self.thread = threading.Thread(target=self._run, name="raw-jsonl-writer", daemon=True)

    def __enter__(self) -> "AsyncJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def write(self, record: dict) -> None:
        if self.error is not None:
            raise RuntimeError("raw writer failed") from self.error
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                self.dropped_count += 1
            else:
                self.dropped_count += 1

    def _run(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                while True:
                    line = self.queue.get()
                    try:
                        if line is None:
                            break
                        handle.write(
                            json.dumps(line, separators=(",", ":"), allow_nan=False)
                            + "\n"
                        )
                    finally:
                        self.queue.task_done()
                handle.flush()
        except BaseException as exc:  # propagate writer-thread failures to caller
            self.error = exc

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("raw writer failed") from self.error

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


def record_stream(
    *,
    host: str,
    port: int,
    output: Path,
    max_records: int = 0,
    max_bytes: int = 10 * 1024**3,
) -> int:
    output = Path(output)
    cache = BoundedCache(output.parent, max_bytes=max_bytes)
    if not cache.enforce(protected=(output,)).within_limit:
        print("[raw-recorder] library is already at the configured limit", flush=True)
        return 2
    records = 0
    started = time.monotonic()
    last_report = started
    with socket.create_connection((host, int(port)), timeout=30.0) as connection:
        connection.settimeout(30.0)
        buffer = b""
        with AsyncJsonlWriter(output) as writer:
            while max_records <= 0 or records < max_records:
                chunk = connection.recv(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if message.get("type") != "raw_state":
                        continue
                    writer.write(normalize_raw_record(message))
                    records += 1
                    now = time.monotonic()
                    if records % 1000 == 0 or now - last_report >= 10.0:
                        report = cache.enforce(protected=(output,))
                        if not report.within_limit:
                            print("[raw-recorder] active file reached the limit; stopping")
                            return 0
                        print(
                            f"[raw-recorder] records={records} recorder_hz="
                            f"{records / max(0.001, now - started):.1f} "
                            f"queue={writer.queue.qsize()}",
                            flush=True,
                        )
                        last_report = now
                    if max_records > 0 and records >= max_records:
                        break
    print(f"[raw-recorder] complete records={records} output={output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4243)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-gib", type=float, default=10.0)
    args = parser.parse_args()
    return record_stream(
        host=args.host,
        port=args.port,
        output=args.output,
        max_records=max(0, args.max_records),
        max_bytes=max(1, int(args.max_gib * 1024**3)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
