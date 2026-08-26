"""Record compact high-frequency snapshots from the Brotato bridge recorder port."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def enforce_library_limit(root: Path, active: Path, max_bytes: int) -> bool:
    """Keep the newest recordings within the library cap.

    The active file is never deleted. If it alone reaches the cap, return
    False so the caller can stop recording cleanly instead of exceeding it.
    """
    files = []
    total = 0
    for candidate in root.rglob("*.jsonl"):
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        total += size
        files.append((candidate.stat().st_mtime, candidate, size))
    if total <= max_bytes:
        return True
    for _, candidate, size in sorted(files):
        if candidate.resolve() == active.resolve():
            continue
        try:
            candidate.unlink()
        except OSError:
            continue
        total -= size
        print(f"[raw-recorder] removed old recording={candidate}", flush=True)
        if total <= max_bytes:
            return True
    return active.stat().st_size < max_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4243)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--max-gib",
        type=float,
        default=10.0,
        help="maximum total JSONL recording-library size (default: 10 GiB)",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = max(1, int(args.max_gib * 1024**3))
    if not enforce_library_limit(args.output.parent, args.output, max_bytes):
        print(f"[raw-recorder] library is already at the {args.max_gib:g} GiB limit")
        return 2
    records = 0
    started = time.monotonic()
    with socket.create_connection((args.host, args.port), timeout=30.0) as connection:
        connection.settimeout(30.0)
        buffer = b""
        with args.output.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
            while args.max_records <= 0 or records < args.max_records:
                chunk = connection.recv(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    message = json.loads(line)
                    if message.get("type") != "raw_state":
                        continue
                    output.write(json.dumps(message, separators=(",", ":")) + "\n")
                    records += 1
                    if records % 1000 == 0:
                        output.flush()
                        if not enforce_library_limit(args.output.parent, args.output, max_bytes):
                            print(
                                f"[raw-recorder] reached {args.max_gib:g} GiB limit; stopping",
                                flush=True,
                            )
                            return 0
                        elapsed = max(0.001, time.monotonic() - started)
                        print(f"[raw-recorder] records={records} hz={records / elapsed:.1f}", flush=True)
    print(f"[raw-recorder] complete records={records} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
