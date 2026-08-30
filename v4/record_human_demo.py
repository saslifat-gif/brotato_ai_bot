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
from v4.bridge_server import BridgeServer
from v4.config import load_config
from v4.protocol import configure_message


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
    parser.add_argument("--run-label", default="", help="human-readable label stored in metadata")
    parser.add_argument("--notes", default="", help="operator notes stored in metadata")
    parser.add_argument("--host", default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    parser.add_argument("--raw-port", type=int, default=4243)
    parser.add_argument("--state-hz", type=float, default=24.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--continue-after-terminal",
        action="store_true",
        help="keep recording after death/victory so F9 bookmarks can cover repeated runs",
    )
    parser.add_argument(
        "--require-capture",
        action="store_true",
        help="return failure if raw samples, rewards, features, or outcomes are missing",
    )
    args = parser.parse_args()
    if not 4.0 <= args.state_hz <= 60.0:
        parser.error("--state-hz must be between 4 and 60")
    if getattr(cfg.policy_mode, "value", str(cfg.policy_mode)) != "HANDCRAFTED":
        parser.error(
            "human-demo recording requires BROTATO_V4_POLICY_MODE=HANDCRAFTED; "
            "learned and hybrid controller modes are refused"
        )

    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing dataset: {output}")
    writer = HumanDemoWriter(output)
    writer.set_metadata("capture_mode", "human_observation_only")
    writer.set_metadata("hybrid_human", "disabled")
    writer.set_metadata("policy_mode", "HANDCRAFTED")
    writer.set_metadata("production_controller_changed", False)
    writer.set_metadata("run_label", args.run_label)
    writer.set_metadata("operator_notes", args.notes)
    writer.set_metadata("configured_rich_state_hz", args.state_hz)
    writer.set_metadata("configured_raw_state_hz", 60.0)
    writer.set_metadata("manual_mark_key", "F9")
    writer.set_metadata("manual_marking", "observation_only; press F9 in game; no controller effect")
    writer.set_metadata("continue_after_terminal", args.continue_after_terminal)
    server = BridgeServer(args.host, args.port)
    stop = threading.Event()
    raw_thread = threading.Thread(
        target=_raw_worker, args=(args.host, args.raw_port, writer, stop),
        name="human-demo-raw-60hz", daemon=True,
    )
    server.start()
    # Do not attach the raw stream to a stale death/victory screen when the
    # recorder is started after a previous run. It begins once the first
    # accepted non-terminal rich frame establishes the new episode boundary.
    raw_thread_started = False
    frames = 0
    last_tick = None
    last_session = ""
    last_phase = ""
    seen_nonterminal = False
    hello_recorded = False
    print(f"[human-demo] output={output} rich_state_hz={args.state_hz:g} raw_state_hz=60")
    print("[human-demo] human controls combat and all build/menu choices; recorder sends nothing")
    print("[human-demo] HYBRID_HUMAN=disabled; production controller is not started")
    print("[human-demo] press F9 to bookmark the exact in-game state; bookmarks may be repeated")
    try:
        while args.max_frames <= 0 or frames < args.max_frames:
            state = server.wait_for_state(timeout_sec=cfg.reset_timeout_sec, after_tick=last_tick)
            last_tick = int(state.get("tick", -1))
            session = str(state.get("session", ""))
            if session != last_session:
                if last_session:
                    writer.finish_episode(outcome="session_boundary", end_phase=last_phase)
                server.send(configure_message(state_hz=args.state_hz))
                last_session = session
                last_phase = ""
                seen_nonterminal = False
                if not hello_recorded and server.last_hello:
                    writer.set_metadata("bridge_hello", server.last_hello)
                    hello_recorded = True
            phase = str(state.get("phase", "unknown"))
            terminal = phase in {"game_over", "victory"}
            # A recorder may be started while the previous run's terminal
            # screen is still visible. Ignore that stale terminal state and
            # wait for the first non-terminal frame of the new run. Once a
            # run is accepted, retain exactly one terminal frame; the optional
            # continuation mode then waits for another manual run.
            if terminal and not seen_nonterminal:
                if state.get("manual_marks"):
                    writer.record_manual_marks_for_last_frame(state)
                last_phase = phase
                continue
            if not terminal:
                seen_nonterminal = True
            if writer.episode_id is None:
                writer.start_episode(phase=phase, source_session_id=session or None)
            if not raw_thread_started:
                raw_thread.start()
                raw_thread_started = True
            writer.record_frame(state, received_ns=time.monotonic_ns())
            frames += 1
            if terminal:
                writer.finish_episode(
                    outcome="victory" if phase == "victory" else "death", end_phase=phase
                )
                if not args.continue_after_terminal:
                    # A recording file represents one complete manual run. Stop
                    # at the first terminal frame so the post-death menu cannot
                    # become a second open/unknown episode.
                    break
                # Keep the terminal screen out of the next episode. If the
                # operator presses F9 repeatedly here, the marker-only path
                # above attaches each new mark to this terminal frame.
                seen_nonterminal = False
                print("[human-demo] terminal recorded; start the next run or press Ctrl+C", flush=True)
                continue
            last_phase = phase
            if frames % 250 == 0:
                print(f"[human-demo] frames={frames} raw_samples=streaming phase={phase}", flush=True)
    except KeyboardInterrupt:
        print(f"[human-demo] stopping after frames={frames}")
    finally:
        stop.set()
        if raw_thread_started:
            raw_thread.join(timeout=2.0)
        server.close()
        writer.finalize()
        writer.close()
    report = validate_dataset(output, require_capture=args.require_capture)
    report_path = output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[human-demo] complete frames={frames} raw_samples={report['raw_samples']} "
        f"manual_marks={report.get('manual_marks', 0)} output={output}"
    )
    status = "PASS" if report.get("capture_ready", report["ok"]) else "FAIL"
    print(
        f"[human-demo] validation={status} errors={len(report['errors'])} "
        f"warnings={len(report['warnings'])} report={report_path}"
    )
    return 0 if report.get("capture_ready", report["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
