"""Refresh the V4 raw-anchor cache without starting a trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from brotato_ai.data.cache import BoundedCache
from v4.train_temporal_hierarchical import load_raw_anchor_arrays


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V4 raw-anchor cache")
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=Path("models/version_3/raw_records"),
    )
    parser.add_argument("--max-records", type=int, default=50_000)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-gib", type=float, default=10.0)
    args = parser.parse_args()
    cache = BoundedCache(
        args.raw_dataset,
        max_bytes=max(1, int(args.max_gib * 1024**3)),
    )
    if not cache.enforce().within_limit:
        print("[raw-cache] dataset exceeds the configured limit", flush=True)
        return 2
    print(
        f"[raw-cache] refreshing dataset={args.raw_dataset} "
        f"max_records={args.max_records} stride={args.stride}",
        flush=True,
    )
    cache.enforce(protected=(args.raw_dataset / "v4_raw_anchor_cache.npz",))
    features, actions = load_raw_anchor_arrays(
        args.raw_dataset,
        max_records=max(0, int(args.max_records)),
        stride=max(1, int(args.stride)),
    )
    print(f"[raw-cache] ready records={len(actions)} features={features.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
