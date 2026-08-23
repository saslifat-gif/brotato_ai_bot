"""Import an OBS recording into the v2 frame-curation workflow."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2


def choose_video() -> Path | None:
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select an OBS Brotato recording",
            filetypes=(
                ("Video files", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()
        return Path(selected).resolve() if selected else None
    except Exception as exc:
        print(f"[obs-import] file picker unavailable: {exc}")
        return None


def source_frame_indices(source_fps: float, target_fps: float, total_frames: int):
    """Yield source indices at approximately the requested output rate."""
    step = max(1.0, float(source_fps) / max(0.1, float(target_fps)))
    next_index = 0.0
    while int(round(next_index)) < total_frames:
        yield int(round(next_index))
        next_index += step


def main() -> int:
    parser = argparse.ArgumentParser(description="Import an OBS video for v2 labeling")
    parser.add_argument("video", nargs="?", help="OBS .mp4/.mkv path; opens a picker when omitted")
    parser.add_argument("--fps", type=float, default=5.0, help="frames to extract per second")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--output", default="datasets/v2/raw")
    args = parser.parse_args()

    video = Path(args.video).resolve() if args.video else choose_video()
    if video is None:
        print("[obs-import] cancelled")
        return 1
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(
            f"could not open {video}; in OBS, remux the recording to MP4 and try again"
        )

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    target_fps = min(source_fps, max(0.1, float(args.fps)))
    quality = int(max(50, min(100, args.jpeg_quality)))

    root = Path(args.output).resolve() / datetime.now().strftime("session_obs_%Y%m%d_%H%M%S")
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sampling_limit = total_frames if total_frames > 0 else 2_147_483_647
    wanted = iter(source_frame_indices(source_fps, target_fps, sampling_limit))
    next_wanted = next(wanted, None)
    source_index = 0
    saved = 0
    width = 0
    height = 0
    print(
        f"[obs-import] video={video.name} source_fps={source_fps:.2f} "
        f"extract_fps={target_fps:.2f}"
    )
    try:
        while next_wanted is not None:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if source_index >= next_wanted:
                height, width = frame.shape[:2]
                image_path = frames_dir / f"frame_{saved:08d}.jpg"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise RuntimeError(f"could not save frame: {image_path}")
                saved += 1
                next_wanted = next(wanted, None)
                if saved % 250 == 0:
                    print(f"[obs-import] extracted {saved} frames...")
            source_index += 1
    finally:
        capture.release()

    if saved == 0:
        raise RuntimeError("the video opened but no frames could be decoded")
    metadata = {
        "source_type": "obs_video",
        "source_video": video.name,
        "source_fps": source_fps,
        "window_title": "Brotato",
        "region": [0, 0, width, height],
        "fps": target_fps,
        "frames": saved,
        "duration_sec": (total_frames or source_index) / max(0.1, source_fps),
        "action_labels": False,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[obs-import] complete session={root} frames={saved}")
    print("[obs-import] next: run curate_v2.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
