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


class DecisionTraceLogger:
    """Append exactly one stable JSON object per control decision."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None

    def record(self, trace: DecisionTrace) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(trace.to_dict(), separators=(",", ":"), allow_nan=False)
                + "\n"
            )


class AsyncJsonlWriter:
    def __init__(self, path: Path, *, queue_size: int = 4096):
        self.path = Path(path)
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=max(16, queue_size))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="raw-jsonl-writer", daemon=True)

    def __enter__(self) -> "AsyncJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def write(self, record: dict) -> None:
        if self.error is not None:
            raise RuntimeError("raw writer failed") from self.error
        self.queue.put(json.dumps(record, separators=(",", ":"), allow_nan=False))

    def _run(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                while True:
                    line = self.queue.get()
                    try:
                        if line is None:
                            break
                        handle.write(line + "\n")
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
