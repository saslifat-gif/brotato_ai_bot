"""CLI for fixed-input v4 temporal history ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.train_temporal_hierarchical import (
    HISTORY_SIZE,
    HumanAnchoredPPO,
    BULLET_HELL_OBSERVATION_SIZE,
    load_raw_anchor_arrays,
)
from brotato_ai.evaluation.temporal_ablation import evaluate_temporal_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("raw_dataset", type=Path)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    features, _ = load_raw_anchor_arrays(
        args.raw_dataset,
        max_records=max(1, args.max_records),
        stride=max(1, args.stride),
        cache_only=False,
    )
    if not len(features):
        raise RuntimeError("no raw observations were available")
    model = HumanAnchoredPPO.load(args.model, device="cpu")
    report = evaluate_temporal_ablation(
        model,
        features,
        history_start=BULLET_HELL_OBSERVATION_SIZE,
        history_size=HISTORY_SIZE,
        seed=args.seed,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
