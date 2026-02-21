import argparse
import csv
import os

import cv2
import numpy as np


WINDOW_NAME = "HP Pixel Labeler"


def parse_args():
    parser = argparse.ArgumentParser(description="Manual/auto pixel-based HP labeling tool.")
    parser.add_argument("--video", default="brotato.mp4", help="Input video path")
    parser.add_argument("--out-dir", default="visual_regression_dataset", help="Output dataset directory")
    parser.add_argument("--left", type=int, default=23, help="HP crop left")
    parser.add_argument("--top", type=int, default=22, help="HP crop top")
    parser.add_argument("--width", type=int, default=319, help="HP crop width")
    parser.add_argument("--height", type=int, default=48, help="HP crop height")
    parser.add_argument("--stride", type=int, default=5, help="Label one frame every N frames")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start from N seconds")
    parser.add_argument("--jump-sec", type=float, default=5.0, help="Seek seconds for J/L keys in manual mode")
    parser.add_argument("--scale", type=int, default=3, help="Preview scale factor in manual mode")
    parser.add_argument("--select-roi", dest="select_roi", action="store_true", help="Select ROI at startup")
    parser.add_argument("--no-select-roi", dest="select_roi", action="store_false", help="Use CLI ROI directly")
    parser.set_defaults(select_roi=True)

    parser.add_argument("--auto-label", action="store_true", help="Auto label sampled frames in selected ROI")
    parser.add_argument("--auto-thresh", type=float, default=0.12, help="Min red-pixel ratio per column")
    parser.add_argument("--auto-jump-limit", type=int, default=40, help="Max px jump between adjacent labeled frames")
    parser.add_argument("--auto-preview", action="store_true", help="Show preview while auto labeling")
    return parser.parse_args()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def ensure_labels_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "ratio", "filled_px", "total_px", "frame_idx", "source"])


def append_label(path: str, image_name: str, ratio: float, filled_px: int, total_px: int, frame_idx: int, source: str):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([image_name, f"{ratio:.6f}", int(filled_px), int(total_px), int(frame_idx), source])


def seek_frame(cap: cv2.VideoCapture, target_frame: int, total_frames: int) -> int:
    max_frame = max(0, total_frames - 1)
    f = clamp(target_frame, 0, max_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    return f


def select_roi_interactive(frame, fallback_roi):
    x, y, w, h = fallback_roi
    roi = cv2.selectROI("Select HP ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select HP ROI")
    rx, ry, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return x, y, w, h
    return int(rx), int(ry), int(rw), int(rh)


def estimate_marker_x(crop: np.ndarray, prev_x: int | None, col_ratio_thresh: float, jump_limit: int) -> int | None:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 80, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    col_ratio = (mask > 0).mean(axis=0)
    valid_cols = np.where(col_ratio >= col_ratio_thresh)[0]

    if valid_cols.size == 0:
        return None

    marker_x = int(valid_cols[-1] + 1)

    if prev_x is not None and jump_limit > 0:
        delta = marker_x - prev_x
        if abs(delta) > jump_limit:
            marker_x = prev_x + (jump_limit if delta > 0 else -jump_limit)

    return max(0, min(marker_x, crop.shape[1]))


def run_auto_label(
    cap: cv2.VideoCapture,
    out_dir: str,
    labels_csv: str,
    total_frames: int,
    fps: float,
    start_frame: int,
    stride: int,
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    col_ratio_thresh: float,
    jump_limit: int,
    preview: bool,
):
    frame_idx = seek_frame(cap, start_frame, total_frames)
    saved_idx = 0
    prev_x = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_idx = frame_idx
        frame_idx += 1

        if current_idx % stride != 0:
            continue

        y0 = clamp(roi_top, 0, frame.shape[0] - 1)
        x0 = clamp(roi_left, 0, frame.shape[1] - 1)
        y1 = clamp(roi_top + roi_height, y0 + 1, frame.shape[0])
        x1 = clamp(roi_left + roi_width, x0 + 1, frame.shape[1])
        crop = frame[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue

        marker_x = estimate_marker_x(crop, prev_x, col_ratio_thresh, jump_limit)
        if marker_x is None:
            continue

        prev_x = marker_x
        crop_w = crop.shape[1]
        ratio = marker_x / float(crop_w)

        image_name = f"hp_auto_{saved_idx:06d}.jpg"
        image_path = os.path.join(out_dir, image_name)
        cv2.imwrite(image_path, crop)
        append_label(labels_csv, image_name, ratio, marker_x, crop_w, current_idx, "auto")
        saved_idx += 1

        if saved_idx % 100 == 0:
            print(f"auto saved={saved_idx}, frame={current_idx}, t={current_idx/fps:.2f}s")

        if preview:
            disp = crop.copy()
            cv2.line(disp, (marker_x, 0), (marker_x, disp.shape[0] - 1), (0, 255, 255), 2)
            cv2.putText(
                disp,
                f"AUTO frame={current_idx} ratio={ratio:.4f}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow(WINDOW_NAME, cv2.resize(disp, (disp.shape[1] * 2, disp.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
            if cv2.waitKey(1) & 0xFF == 27:
                print("auto stopped by Esc")
                break

    cv2.destroyAllWindows()
    print(f"auto done. saved={saved_idx}, labels={labels_csv}")


def run_manual_label(
    cap: cv2.VideoCapture,
    out_dir: str,
    labels_csv: str,
    total_frames: int,
    fps: float,
    start_frame: int,
    stride: int,
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    jump_sec: float,
    scale: int,
):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    ui_state = {
        "scale": max(1, int(scale)),
        "marker_x": roi_width,
        "crop_w": roi_width,
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        s = ui_state["scale"]
        cw = ui_state["crop_w"]
        ui_state["marker_x"] = clamp(int(round(x / float(s))), 0, cw)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    frame_idx = seek_frame(cap, start_frame, total_frames)
    saved_idx = 0
    seek_target = None
    jump_frames = max(1, int(round(jump_sec * fps)))

    while True:
        if seek_target is not None:
            frame_idx = seek_frame(cap, seek_target, total_frames)
            seek_target = None

        ok, frame = cap.read()
        if not ok:
            break

        current_idx = frame_idx
        frame_idx += 1

        if current_idx % stride != 0:
            continue

        y0 = clamp(roi_top, 0, frame.shape[0] - 1)
        x0 = clamp(roi_left, 0, frame.shape[1] - 1)
        y1 = clamp(roi_top + roi_height, y0 + 1, frame.shape[0])
        x1 = clamp(roi_left + roi_width, x0 + 1, frame.shape[1])
        crop = frame[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        ui_state["crop_w"] = crop_w
        ui_state["marker_x"] = clamp(ui_state["marker_x"], 0, crop_w)

        while True:
            marker_x = ui_state["marker_x"]
            disp = crop.copy()

            if marker_x > 0:
                cv2.rectangle(disp, (0, 0), (marker_x - 1, crop_h - 1), (0, 255, 0), 1)
            cv2.line(disp, (marker_x, 0), (marker_x, crop_h - 1), (0, 255, 255), 2)

            ratio = marker_x / float(crop_w)
            t_sec = current_idx / fps
            cv2.putText(
                disp,
                f"frame={current_idx} t={t_sec:.2f}s ratio={ratio:.4f} px={marker_x}/{crop_w}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            s = ui_state["scale"]
            preview = cv2.resize(disp, (crop_w * s, crop_h * s), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):
                image_name = f"hp_manual_{saved_idx:06d}.jpg"
                image_path = os.path.join(out_dir, image_name)
                cv2.imwrite(image_path, crop)
                append_label(labels_csv, image_name, ratio, marker_x, crop_w, current_idx, "manual")
                saved_idx += 1
                print(f"manual saved #{saved_idx}: frame={current_idx}, t={t_sec:.2f}s, ratio={ratio:.4f}")
                break
            if key == ord("s"):
                break
            if key == ord("l"):
                seek_target = current_idx + jump_frames
                break
            if key == ord("j"):
                seek_target = current_idx - jump_frames
                break
            if key == ord("r"):
                roi_left, roi_top, roi_width, roi_height = select_roi_interactive(
                    frame, (roi_left, roi_top, roi_width, roi_height)
                )
                print(f"selected_roi=(left={roi_left}, top={roi_top}, w={roi_width}, h={roi_height})")
                break
            if key == 27:
                cv2.destroyAllWindows()
                print(f"manual done. saved={saved_idx}, labels={labels_csv}")
                return

    cv2.destroyAllWindows()
    print(f"manual done. saved={saved_idx}, labels={labels_csv}")


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels_csv = os.path.join(args.out_dir, "labels.csv")
    ensure_labels_csv(labels_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    start_by_sec = int(round(args.start_sec * fps)) if args.start_sec > 0 else 0
    start_frame = max(int(args.start), start_by_sec)

    roi_left, roi_top, roi_width, roi_height = args.left, args.top, args.width, args.height

    print(f"video={args.video}, total_frames={total_frames}, fps={fps:.2f}")
    print(f"roi=(left={roi_left}, top={roi_top}, w={roi_width}, h={roi_height}), stride={args.stride}")
    print(f"start_frame={start_frame} (start={args.start}, start_sec={args.start_sec})")

    if args.select_roi:
        fidx = seek_frame(cap, start_frame, total_frames)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read start frame for ROI selection.")
        roi_left, roi_top, roi_width, roi_height = select_roi_interactive(
            first_frame, (roi_left, roi_top, roi_width, roi_height)
        )
        print(f"selected_roi=(left={roi_left}, top={roi_top}, w={roi_width}, h={roi_height})")
        seek_frame(cap, fidx, total_frames)

    if args.auto_label:
        print("mode=auto, source=auto")
        run_auto_label(
            cap=cap,
            out_dir=args.out_dir,
            labels_csv=labels_csv,
            total_frames=total_frames,
            fps=fps,
            start_frame=start_frame,
            stride=args.stride,
            roi_left=roi_left,
            roi_top=roi_top,
            roi_width=roi_width,
            roi_height=roi_height,
            col_ratio_thresh=args.auto_thresh,
            jump_limit=args.auto_jump_limit,
            preview=args.auto_preview,
        )
    else:
        print("mode=manual, source=manual")
        print("Control: Mouse left click = set bar end x")
        print("Keys: Space/Enter save, S skip, J back, L forward, R reselect ROI, Esc quit")
        run_manual_label(
            cap=cap,
            out_dir=args.out_dir,
            labels_csv=labels_csv,
            total_frames=total_frames,
            fps=fps,
            start_frame=start_frame,
            stride=args.stride,
            roi_left=roi_left,
            roi_top=roi_top,
            roi_width=roi_width,
            roi_height=roi_height,
            jump_sec=args.jump_sec,
            scale=args.scale,
        )

    cap.release()


if __name__ == "__main__":
    main()
