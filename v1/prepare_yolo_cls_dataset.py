import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def iter_images(folder: Path):
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def iter_videos(folder: Path):
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def extract_frames_from_video(
    video_path: Path,
    frame_stride: int,
    max_frames: int,
    start_sec: float,
    end_sec: float,
    progress_every: int = 0,
    progress_prefix: str = "",
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(start_sec * fps))
    end_frame = total_frames - 1
    if end_sec > 0:
        end_frame = min(end_frame, int(end_sec * fps))
    if total_frames <= 0:
        start_frame = 0
        end_frame = 10**9

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out: list[np.ndarray] = []
    idx = start_frame
    saved = 0
    stride = max(1, int(frame_stride))
    report_step = max(0, int(progress_every))
    prefix = progress_prefix.strip()
    if prefix:
        prefix = f"{prefix} "
    print(
        f"[video] {prefix}open={video_path.name} fps={fps:.2f} "
        f"start={start_frame} end={end_frame} stride={stride} max={max_frames}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx > end_frame:
            break
        if idx % stride == 0:
            out.append(frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        if report_step > 0:
            scanned = idx - start_frame + 1
            if scanned % report_step == 0:
                print(f"[video] {prefix}{video_path.name} scanned={scanned} saved={saved}")
        idx += 1

    cap.release()
    print(f"[video] {prefix}done {video_path.name} saved={saved}")
    return out


def dhash(img: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedup_images(imgs: list[np.ndarray], ham_threshold: int) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    hashes: list[int] = []
    for img in imgs:
        h = dhash(img)
        if any(hamming(h, old) <= ham_threshold for old in hashes):
            continue
        hashes.append(h)
        kept.append(img)
    return kept


def augment_one(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    tx = random.uniform(-0.04, 0.04) * w
    ty = random.uniform(-0.04, 0.04) * h
    scale = random.uniform(0.95, 1.05)
    angle = random.uniform(-4.0, 4.0)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += [tx, ty]
    out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    alpha = random.uniform(0.85, 1.20)
    beta = random.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Prepare deduplicated frame dataset for YOLO classification (YOLO-CLS)."
    )
    parser.add_argument("--src", default="dataset_states")
    parser.add_argument("--dst", default="dataset_yolo_cls")
    parser.add_argument("--dedup-hamming", type=int, default=2)
    parser.add_argument("--video-frame-stride", type=int, default=4)
    parser.add_argument("--video-max-frames", type=int, default=2500)
    parser.add_argument("--video-start-sec", type=float, default=0.0)
    parser.add_argument("--video-end-sec", type=float, default=0.0)
    parser.add_argument("--video-progress-every", type=int, default=800)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=0,
        help="0 means keep unique count as-is; >0 means force every class to this count",
    )
    parser.add_argument(
        "--fill-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When target-per-class is set and class is short, fill with light augmentation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_absolute():
        src = (base_dir / src).resolve()
    if not dst.is_absolute():
        dst = (base_dir / dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    classes = [d for d in src.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found in {src}")

    print("Preparing YOLO-CLS dataset...")
    for cls in classes:
        imgs: list[np.ndarray] = []
        n_img_files = 0
        n_video_files = 0

        for p in iter_images(cls):
            n_img_files += 1
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)

        for vp in iter_videos(cls):
            n_video_files += 1
            frames = extract_frames_from_video(
                vp,
                frame_stride=args.video_frame_stride,
                max_frames=args.video_max_frames,
                start_sec=args.video_start_sec,
                end_sec=args.video_end_sec,
                progress_every=args.video_progress_every,
                progress_prefix=f"{cls.name}:",
            )
            imgs.extend(frames)

        if not imgs:
            print(f"[skip] {cls.name}: no images/videos")
            continue

        unique = dedup_images(imgs, ham_threshold=args.dedup_hamming)
        selected = unique
        capped = False
        filled = False

        if args.target_per_class > 0 and len(unique) > args.target_per_class:
            pick = np.random.choice(len(unique), size=args.target_per_class, replace=False)
            selected = [unique[int(i)] for i in pick]
            capped = True

        out_dir = dst / cls.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.iterdir():
            if old.is_file():
                old.unlink()

        count = 0
        for i, img in enumerate(selected):
            cv2.imwrite(str(out_dir / f"{cls.name}_u_{i:05d}.jpg"), img)
            count += 1

        if args.target_per_class > 0 and args.fill_augment and count < args.target_per_class and len(selected) > 0:
            filled = True
            while count < args.target_per_class:
                base = selected[random.randrange(len(selected))]
                aug = augment_one(base)
                cv2.imwrite(str(out_dir / f"{cls.name}_a_{count:05d}.jpg"), aug)
                count += 1

        print(
            f"[ok] {cls.name}: img_files={n_img_files} video_files={n_video_files} "
            f"raw={len(imgs)} unique={len(unique)} out={count} "
            f"cap={'yes' if capped else 'no'} fill={'yes' if filled else 'no'}"
        )

    print(f"Done. Output -> {dst}")


if __name__ == "__main__":
    main()
