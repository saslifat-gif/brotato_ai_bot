"""Record rich human gameplay demonstrations for BC, IRL, and replay analysis.

The game remains under human control in combat and in menus.  The recorder
opens the existing rich-state channel and the independent 60 Hz raw channel;
it never sends movement or UI actions.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

from brotato_ai.data.human_demo import HumanDemoWriter, validate_dataset
from v3.bridge_server import BridgeServer
from v3.config import load_config
from v3.protocol import configure_message


def _raw_worker(host: str, port: int, writer: HumanDemoWriter, stop: threading.Event) -> None:
    """Copy the bridge's 60 Hz stream without blocking the rich-state loop."""

    while not stop.is_set():
        try:
            with socket.create_connection((host, int(port)), timeout=3.0) as connection:
                connection.settimeout(1.0)
                buffer = b""
                while not stop.is_set():
                    try:
                        chunk = connection.recv(1024 * 1024)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if payload.get("type") == "raw_state":
                            writer.record_raw_sample(payload)
        except (OSError, TimeoutError):
            stop.wait(0.5)


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Record rich human gameplay demonstrations")
    parser.add_argument("--output", type=Path, required=True, help="SQLite demonstration dataset")
    parser.add_argument("--host", default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    parser.add_argument("--raw-port", type=int, default=4243)
    parser.add_argument("--state-hz", type=float, default=24.0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if not 4.0 <= args.state_hz <= 60.0:
        parser.error("--state-hz must be between 4 and 60")

    output = args.output.resolve()
    writer = HumanDemoWriter(output)
    server = BridgeServer(args.host, args.port)
    stop = threading.Event()
    raw_thread = threading.Thread(
        target=_raw_worker, args=(args.host, args.raw_port, writer, stop),
        name="human-demo-raw-60hz", daemon=True,
    )
    server.start()
    raw_thread.start()
    frames = 0
    last_tick = None
    last_session = ""
    last_phase = ""
    print(f"[human-demo] output={output} rich_state_hz={args.state_hz:g} raw_state_hz=60")
    print("[human-demo] human controls combat and all build/menu choices; recorder sends nothing")
    try:
        while args.max_frames <= 0 or frames < args.max_frames:
            state = server.wait_for_state(timeout_sec=cfg.reset_timeout_sec, after_tick=last_tick)
            last_tick = int(state.get("tick", -1))
            session = str(state.get("session", ""))
            if session != last_session:
                if last_session:
                    writer.finish_episode(outcome="session_boundary", end_phase=last_phase)
                server.send(configure_message(state_hz=args.state_hz))
                writer.start_episode(phase=str(state.get("phase", "unknown")))
                last_session = session
                last_phase = ""
            phase = str(state.get("phase", "unknown"))
            writer.record_frame(state, received_ns=time.monotonic_ns())
            frames += 1
            if phase in {"game_over", "victory"} and last_phase == "combat":
                writer.finish_episode(
                    outcome="victory" if phase == "victory" else "death", end_phase=phase
                )
                writer.start_episode(phase=phase)
            last_phase = phase
            if frames % 250 == 0:
                print(f"[human-demo] frames={frames} raw_samples=streaming phase={phase}", flush=True)
    except KeyboardInterrupt:
        print(f"[human-demo] stopping after frames={frames}")
    finally:
        stop.set()
        raw_thread.join(timeout=2.0)
        server.close()
        writer.finalize()
        writer.close()
    report = validate_dataset(output)
    print(f"[human-demo] complete frames={frames} raw_samples={report['raw_samples']} output={output}")
    print(f"[human-demo] validation={'PASS' if report['ok'] else 'FAIL'} errors={len(report['errors'])}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
