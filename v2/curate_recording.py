"""Interactively sort recorded frames into combat and UI labeling queues."""

import argparse
import shutil
from pathlib import Path

import cv2


def latest_session(raw_root: Path) -> Path:
    sessions = sorted(
        (path for path in raw_root.glob("session_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        raise RuntimeError(f"no recorded sessions found under {raw_root}")
    return sessions[0]


def copy_selected(frame_path: Path, destination: Path, session_name: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{session_name}_{frame_path.name}"
    shutil.copy2(frame_path, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate v2 recorded frames")
    parser.add_argument("--session", default="latest", help="session directory or 'latest'")
    parser.add_argument("--raw-root", default="datasets/v2/raw")
    parser.add_argument("--output", default="datasets/v2/to_label")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    session = latest_session(raw_root) if args.session == "latest" else Path(args.session).resolve()
    frames = sorted((session / "frames").glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"no frames found under {session / 'frames'}")

    output = Path(args.output).resolve()
    combat_dir = output / "combat"
    ui_dir = output / "ui"
    stride = max(1, int(args.stride))
    index = 0
    combat_count = 0
    ui_count = 0
    print(f"[curate] session={session} frames={len(frames)} stride={stride}")
    print("[curate] C=combat U=UI S/Space=skip N=next frame P=previous frame Q/Esc=quit")

    try:
        while 0 <= index < len(frames):
            frame_path = frames[index]
            canvas = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if canvas is None:
                index += stride
                continue
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 78), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                f"{index + 1}/{len(frames)}  {frame_path.name}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                "C combat | U UI | S skip | N/P one frame | Q quit",
                (16, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )
            cv2.imshow("Brotato v2 Frame Curation", canvas)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                copy_selected(frame_path, combat_dir, session.name)
                combat_count += 1
                index += stride
            elif key in (ord("u"), ord("U")):
                copy_selected(frame_path, ui_dir, session.name)
                ui_count += 1
                index += stride
            elif key in (ord("s"), ord("S"), ord(" ")):
                index += stride
            elif key in (ord("n"), ord("N")):
                index += 1
            elif key in (ord("p"), ord("P")):
                index = max(0, index - 1)
    finally:
        cv2.destroyAllWindows()

    print(f"[curate] selected this run combat={combat_count} ui={ui_count}")
    print(f"[curate] combat queue={combat_dir}")
    print(f"[curate] ui queue={ui_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
